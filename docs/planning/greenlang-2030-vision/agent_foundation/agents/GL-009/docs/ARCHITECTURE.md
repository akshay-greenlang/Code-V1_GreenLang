# GL-009 THERMALIQ - System Architecture Documentation

**Agent:** GL-009 ThermalEfficiencyCalculator
**Version:** 1.0.0
**Domain:** Process Heat Systems
**Status:** Production Ready
**Last Updated:** 2025-11-26

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Principles](#architecture-principles)
4. [Component Architecture](#component-architecture)
5. [Core Components](#core-components)
6. [Deterministic Calculator Suite](#deterministic-calculator-suite)
7. [Integration Connectors](#integration-connectors)
8. [Visualization Engine](#visualization-engine)
9. [Data Flow Architecture](#data-flow-architecture)
10. [Physics Formulas (Zero-Hallucination)](#physics-formulas-zero-hallucination)
11. [Tool Specifications](#tool-specifications)
12. [Performance Architecture](#performance-architecture)
13. [Security Architecture](#security-architecture)
14. [Deployment Architecture](#deployment-architecture)
15. [Scalability & High Availability](#scalability--high-availability)
16. [Technology Stack](#technology-stack)
17. [Design Patterns](#design-patterns)
18. [Error Handling Strategy](#error-handling-strategy)
19. [Monitoring & Observability](#monitoring--observability)
20. [Future Enhancements](#future-enhancements)

---

## Executive Summary

GL-009 THERMALIQ is a production-grade autonomous agent for calculating overall thermal efficiency of industrial heat processes with zero-hallucination guarantees. The architecture implements a deterministic, physics-based approach using First Law and Second Law thermodynamic analysis, complemented by Sankey diagram visualization for energy flow understanding.

**Key Architectural Highlights:**

- **Zero-Hallucination Design**: All calculations use deterministic physics equations (First Law, Second Law, Napier)
- **Comprehensive Heat Analysis**: First Law efficiency, Exergy efficiency, heat loss breakdown
- **Visual Energy Flows**: Sankey diagram generation for intuitive energy balance visualization
- **Industry Standards Compliance**: ASME PTC 4, ISO 50001, DOE AMO guidelines
- **Real-Time Integration**: OPC-UA, Modbus, SCADA, and process historian connectivity
- **Provenance Tracking**: Complete SHA-256 audit trail for regulatory compliance
- **Cost-Optimized**: Deterministic calculations only, LLM for classification/reporting only

---

## System Overview

### High-Level Architecture Diagram (ASCII)

```
+==============================================================================+
|                      GL-009 THERMALIQ ARCHITECTURE                           |
+==============================================================================+
|                                                                              |
|  +------------------+    +-------------------+    +---------------------+    |
|  |  Energy Meters   |    |  Process          |    |  SCADA/DCS          |    |
|  |  (Modbus/OPC-UA) |--->|  Historians       |--->|  Real-time Data     |    |
|  +------------------+    | (PI, Wonderware)  |    +---------------------+    |
|           |              +-------------------+              |                |
|           |                       |                         |                |
|           v                       v                         v                |
|  +------------------------------------------------------------------------+  |
|  |                        DATA INTAKE LAYER                               |  |
|  |  +----------------+  +----------------+  +----------------+            |  |
|  |  | Energy Meter   |  | Historian      |  | SCADA          |            |  |
|  |  | Connector      |  | Connector      |  | Connector      |            |  |
|  |  +----------------+  +----------------+  +----------------+            |  |
|  +------------------------------------------------------------------------+  |
|                                    |                                         |
|                                    v                                         |
|  +------------------------------------------------------------------------+  |
|  |                      VALIDATION & NORMALIZATION                        |  |
|  |  +------------------+  +------------------+  +------------------+      |  |
|  |  | Schema Validator |  | Unit Converter   |  | Outlier Detector |      |  |
|  |  +------------------+  +------------------+  +------------------+      |  |
|  +------------------------------------------------------------------------+  |
|                                    |                                         |
|                                    v                                         |
|  +------------------------------------------------------------------------+  |
|  |                  THERMAL EFFICIENCY ORCHESTRATOR                       |  |
|  |                     (ThermalEfficiencyOrchestrator)                    |  |
|  |                                                                        |  |
|  |   Operation Modes: calculate | analyze | benchmark | visualize | report|  |
|  |   Thread-Safe Cache: TTL-based, 85%+ hit rate target                   |  |
|  |   Provenance: SHA-256 hash chain                                       |  |
|  +------------------------------------------------------------------------+  |
|                                    |                                         |
|       +----------------------------+----------------------------+            |
|       |                            |                            |            |
|       v                            v                            v            |
|  +-----------+              +-----------+              +-----------+         |
|  | FIRST LAW |              | SECOND LAW|              | HEAT LOSS |         |
|  | EFFICIENCY|              | (EXERGY)  |              | ANALYSIS  |         |
|  | CALCULATOR|              | CALCULATOR|              | MODULE    |         |
|  +-----------+              +-----------+              +-----------+         |
|       |                            |                            |            |
|       +----------------------------+----------------------------+            |
|                                    |                                         |
|       +----------------------------+----------------------------+            |
|       |                            |                            |            |
|       v                            v                            v            |
|  +-----------+              +-----------+              +-----------+         |
|  | FUEL      |              | STEAM     |              | ELECTRICAL|         |
|  | ENERGY    |              | ENERGY    |              | ENERGY    |         |
|  | CALCULATOR|              | CALCULATOR|              | CALCULATOR|         |
|  +-----------+              +-----------+              +-----------+         |
|       |                            |                            |            |
|       +----------------------------+----------------------------+            |
|                                    |                                         |
|                                    v                                         |
|  +------------------------------------------------------------------------+  |
|  |                    ANALYSIS & OPTIMIZATION LAYER                       |  |
|  |  +------------------+  +------------------+  +------------------+      |  |
|  |  | Benchmark        |  | Improvement      |  | Uncertainty      |      |  |
|  |  | Calculator       |  | Analyzer         |  | Quantifier       |      |  |
|  |  +------------------+  +------------------+  +------------------+      |  |
|  +------------------------------------------------------------------------+  |
|                                    |                                         |
|                                    v                                         |
|  +------------------------------------------------------------------------+  |
|  |                      VISUALIZATION ENGINE                              |  |
|  |  +------------------+  +------------------+  +------------------+      |  |
|  |  | Sankey Diagram   |  | Waterfall Chart  |  | Trend Analysis   |      |  |
|  |  | Generator        |  | Generator        |  | Visualizer       |      |  |
|  |  +------------------+  +------------------+  +------------------+      |  |
|  +------------------------------------------------------------------------+  |
|                                    |                                         |
|                                    v                                         |
|  +------------------------------------------------------------------------+  |
|  |                        OUTPUT LAYER                                    |  |
|  |  +------------------+  +------------------+  +------------------+      |  |
|  |  | JSON/API         |  | PDF Report       |  | CSV Export       |      |  |
|  |  | Response         |  | Generator        |  | Module           |      |  |
|  |  +------------------+  +------------------+  +------------------+      |  |
|  +------------------------------------------------------------------------+  |
|                                                                              |
+==============================================================================+
```

### Component Relationships

```
                    +---------------------------+
                    |    External Data Sources  |
                    | (Meters, Historians, SCADA)|
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |   Integration Connectors  |
                    | EnergyMeter | Historian   |
                    | SCADA | ERP               |
                    +-------------+-------------+
                                  |
                                  v
+-------------------+   +-------------------+   +-------------------+
|  Data Validation  |-->|   Orchestrator    |-->|  Provenance       |
|  & Normalization  |   | (Thread-Safe)     |   |  Tracker          |
+-------------------+   +-------------------+   +-------------------+
                                  |
        +------------+------------+------------+------------+
        |            |            |            |            |
        v            v            v            v            v
+----------+  +----------+  +----------+  +----------+  +----------+
|First Law |  |Second Law|  |Heat Loss |  |Fuel Calc |  |Steam Calc|
|Calculator|  |Calculator|  |Calculator|  |(HHV/LHV) |  |(Enthalpy)|
+----------+  +----------+  +----------+  +----------+  +----------+
        |            |            |            |            |
        +------------+------------+------------+------------+
                                  |
                                  v
                    +---------------------------+
                    |   Analysis & Benchmarking |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |   Visualization Engine    |
                    |   (Sankey, Waterfall)     |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    |      Report Generator     |
                    +---------------------------+
```

### Data Flow Patterns

**Pattern 1: Real-Time Efficiency Calculation**
```
Energy Meters (1s) --> Validation --> Calculator --> Cache --> API Response (<500ms)
```

**Pattern 2: Historical Analysis**
```
Historian Query --> Batch Load --> Aggregation --> Trend Analysis --> Report
```

**Pattern 3: Benchmark Comparison**
```
Current Data --> Normalize --> Industry Database Lookup --> Gap Analysis --> Recommendations
```

**Pattern 4: Sankey Visualization**
```
Energy Balance --> Loss Breakdown --> Plotly Generation --> SVG/PNG Output (<2s)
```

---

## Architecture Principles

### 1. Determinism by Default

**Principle**: Same inputs produce exactly same outputs, always.

**Implementation**:
- Physics equations with explicit decimal precision (4 decimal places)
- LLM temperature=0.0, seed=42 for any classification tasks
- Deterministic random number generation for Monte Carlo uncertainty analysis
- Immutable emission factors and heating values

**Verification**:
```python
result1 = agent.calculate_efficiency(energy_data, seed=42)
result2 = agent.calculate_efficiency(energy_data, seed=42)
assert result1 == result2  # Byte-exact match guaranteed
```

### 2. Physics Over Heuristics

**Principle**: Use validated thermodynamic equations, not ML approximations.

**First Law Efficiency**:
```python
# Energy balance: Q_in = Q_useful + Q_losses
eta_1 = Q_useful / Q_in * 100  # Percentage
```

**Second Law (Exergy) Efficiency**:
```python
# Exergy = H - T0 * S (available work)
eta_2 = Exergy_out / Exergy_in * 100  # Percentage
```

**Never**:
- ML models for thermodynamic calculations
- Heuristic multipliers without scientific basis
- Approximations that violate conservation laws

### 3. Conservation Law Enforcement

**Principle**: Energy balance must close within tolerance.

**Validation Rule**:
```python
# Sankey diagram validation
total_input = sum(energy_inputs)
total_output = sum(useful_outputs) + sum(losses)
closure_error = abs(total_input - total_output) / total_input * 100

assert closure_error < 2.0, "Energy balance closure error exceeds 2%"
```

### 4. Fail-Safe Degradation

**Principle**: If a component fails, system continues with reduced functionality.

**Degradation Hierarchy**:
1. **Full Analysis**: All calculators + visualization + benchmarking
2. **Core Efficiency**: First/Second Law calculators only
3. **Basic Calculation**: First Law efficiency only
4. **Manual Fallback**: Return cached last-known-good values
5. **Error State**: Alert operator, no calculation

### 5. Zero Trust Security

**Principle**: Never trust data, validate everything.

**Security Layers**:
- Input validation (type, range, unit checks)
- Secrets management (zero hardcoded credentials)
- Network egress control (allowlist-only)
- Encryption at rest and in transit
- RBAC for all operations
- Audit logging for compliance

---

## Component Architecture

### Core Components Diagram

```
+===========================================================================+
|                    ThermalEfficiencyOrchestrator                          |
|                         (Main Agent Class)                                 |
+===========================================================================+
|                                                                           |
|  +-------------------+    +-------------------+    +-------------------+  |
|  |  Operation Mode   |    |  Thread-Safe      |    |  Provenance       |  |
|  |  Controller       |    |  Cache Manager    |    |  Tracker          |  |
|  |                   |    |  (TTL: 60s)       |    |  (SHA-256)        |  |
|  +-------------------+    +-------------------+    +-------------------+  |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    CALCULATOR SUITE (10+)                          |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |  | FirstLaw      |  | SecondLaw     |  | HeatLoss      |           |   |
|  |  | Efficiency    |  | Efficiency    |  | Calculator    |           |   |
|  |  | Calculator    |  | Calculator    |  | (Radiation,   |           |   |
|  |  | (eta=Qout/Qin)|  | (eta=Ex_out/  |  |  Convection,  |           |   |
|  |  |               |  |  Ex_in)       |  |  Conduction)  |           |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |                                                                    |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |  | FuelEnergy    |  | SteamEnergy   |  | Electrical    |           |   |
|  |  | Calculator    |  | Calculator    |  | Energy        |           |   |
|  |  | (HHV/LHV)     |  | (Enthalpy,    |  | Calculator    |           |   |
|  |  |               |  |  Steam Tables)|  | (Motor/Pump)  |           |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |                                                                    |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |  | Benchmark     |  | Improvement   |  | Uncertainty   |           |   |
|  |  | Calculator    |  | Analyzer      |  | Quantifier    |           |   |
|  |  | (Industry     |  | (Optimization |  | (Measurement  |           |   |
|  |  |  Comparison)  |  |  Opportunities|  |  Uncertainty) |           |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |                                                                    |   |
|  |  +---------------+                                                 |   |
|  |  | Sankey        |                                                 |   |
|  |  | Generator     |                                                 |   |
|  |  | (Energy Flow  |                                                 |   |
|  |  |  Visualization|                                                 |   |
|  |  +---------------+                                                 |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    INTEGRATION CONNECTORS                          |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |  | EnergyMeter   |  | Historian     |  | SCADA         |           |   |
|  |  | Connector     |  | Connector     |  | Connector     |           |   |
|  |  | (Modbus/      |  | (OSIsoft PI,  |  | (Real-time    |           |   |
|  |  |  OPC-UA)      |  |  Wonderware)  |  |  Process)     |           |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |                                                                    |   |
|  |  +---------------+                                                 |   |
|  |  | ERP           |                                                 |   |
|  |  | Connector     |                                                 |   |
|  |  | (SAP/Oracle   |                                                 |   |
|  |  |  Cost Data)   |                                                 |   |
|  |  +---------------+                                                 |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    VISUALIZATION ENGINE                            |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |  | Sankey        |  | Waterfall     |  | Trend         |           |   |
|  |  | Diagram       |  | Chart         |  | Analysis      |           |   |
|  |  | (Plotly)      |  | (Matplotlib)  |  | (Time Series) |           |   |
|  |  +---------------+  +---------------+  +---------------+           |   |
|  |                                                                    |   |
|  |  +---------------+                                                 |   |
|  |  | Loss Breakdown|                                                 |   |
|  |  | Pie Chart     |                                                 |   |
|  |  | (Component %) |                                                 |   |
|  |  +---------------+                                                 |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```

---

## Core Components

### 2.1 Thermal Efficiency Orchestrator

**File**: `thermal_efficiency_orchestrator.py`

**Main Class**: `ThermalEfficiencyOrchestrator(BaseAgent)`

**Responsibilities**:
- Coordinate all thermal efficiency calculations
- Manage operation modes (calculate, analyze, benchmark, visualize, report)
- Implement thread-safe caching with TTL
- Track provenance with SHA-256 hash chains
- Handle graceful degradation on component failures

**Operation Modes**:

| Mode | Description | Output |
|------|-------------|--------|
| `calculate` | Core efficiency calculation | First/Second Law efficiency values |
| `analyze` | Detailed loss breakdown | Heat loss by category (radiation, convection, flue) |
| `benchmark` | Industry comparison | Efficiency vs. industry average, percentile ranking |
| `visualize` | Generate diagrams | Sankey, waterfall, pie charts |
| `report` | Comprehensive report | PDF/HTML with all analyses |

**Thread-Safe Caching**:
```python
class ThermalEfficiencyOrchestrator(BaseAgent):
    def __init__(self, config: ThermalEfficiencyConfig):
        super().__init__(config)

        # Thread-safe cache with TTL
        self._results_cache = ThreadSafeCache(
            max_size=500,
            ttl_seconds=60.0
        )

        # Provenance tracking
        self.provenance_chain = ProvenanceChain(
            algorithm="sha256",
            retention_days=2555  # 7 years for regulatory compliance
        )
```

**Provenance Tracking**:
```python
def _calculate_provenance_hash(
    self,
    input_data: Dict[str, Any],
    result: Dict[str, Any]
) -> str:
    """
    Calculate SHA-256 provenance hash for complete audit trail.

    DETERMINISM GUARANTEE: Identical inputs produce identical hashes.
    """
    # Serialize deterministically (sorted keys)
    input_str = json.dumps(input_data, sort_keys=True, default=str)
    result_str = json.dumps(result, sort_keys=True, default=str)

    provenance_str = f"{self.config.agent_id}|{input_str}|{result_str}"
    hash_value = hashlib.sha256(provenance_str.encode()).hexdigest()

    return hash_value
```

---

## Deterministic Calculator Suite

### Calculator Suite Overview (10+ Calculators)

| # | Calculator | File | Purpose | Key Formula |
|---|------------|------|---------|-------------|
| 1 | First Law Efficiency | `first_law_efficiency_calculator.py` | Energy balance | eta = Q_out / Q_in |
| 2 | Second Law Efficiency | `second_law_efficiency_calculator.py` | Exergy efficiency | eta = Ex_out / Ex_in |
| 3 | Heat Loss | `heat_loss_calculator.py` | Loss breakdown | Q_loss = Q_rad + Q_conv + Q_cond + Q_flue |
| 4 | Sankey Generator | `sankey_generator.py` | Energy flow diagram | Visual balance |
| 5 | Benchmark | `benchmark_calculator.py` | Industry comparison | Percentile ranking |
| 6 | Improvement Analyzer | `improvement_analyzer.py` | Optimization opportunities | ROI ranking |
| 7 | Uncertainty Quantifier | `uncertainty_quantifier.py` | Measurement uncertainty | Monte Carlo (deterministic seed) |
| 8 | Fuel Energy | `fuel_energy_calculator.py` | Fuel heating values | HHV/LHV lookup |
| 9 | Steam Energy | `steam_energy_calculator.py` | Steam enthalpy | IAPWS-IF97 steam tables |
| 10 | Electrical Energy | `electrical_energy_calculator.py` | Motor/pump efficiency | Motor curve interpolation |

### 1. First Law Efficiency Calculator

**File**: `calculators/first_law_efficiency_calculator.py`

**Purpose**: Calculate thermal efficiency based on First Law of Thermodynamics (energy conservation)

**Formula**:
```
eta_1 = (Useful Heat Output) / (Total Energy Input) x 100%

Where:
- Useful Heat Output = Energy delivered to process (kW or GJ)
- Total Energy Input = Fuel energy + Electrical energy (kW or GJ)
```

**Implementation**:
```python
class FirstLawEfficiencyCalculator:
    """
    Deterministic First Law thermal efficiency calculator.

    Physics Basis: First Law of Thermodynamics (Energy Conservation)
    Standards: ASME PTC 4, ISO 50001
    """

    def calculate(
        self,
        energy_input_kw: float,
        useful_output_kw: float,
        losses_kw: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate First Law efficiency.

        Args:
            energy_input_kw: Total energy input (kW)
            useful_output_kw: Useful energy output (kW)
            losses_kw: Optional breakdown of losses

        Returns:
            Efficiency result with provenance
        """
        # Input validation
        assert energy_input_kw > 0, "Energy input must be positive"
        assert useful_output_kw >= 0, "Useful output cannot be negative"
        assert useful_output_kw <= energy_input_kw, "Output cannot exceed input"

        # First Law efficiency calculation
        efficiency_percent = (useful_output_kw / energy_input_kw) * 100

        # Calculate total losses
        if losses_kw:
            total_losses_kw = sum(losses_kw.values())
        else:
            total_losses_kw = energy_input_kw - useful_output_kw

        # Energy balance closure check
        balance_error_percent = abs(
            energy_input_kw - useful_output_kw - total_losses_kw
        ) / energy_input_kw * 100

        return {
            'efficiency_percent': round(efficiency_percent, 2),
            'energy_input_kw': round(energy_input_kw, 2),
            'useful_output_kw': round(useful_output_kw, 2),
            'total_losses_kw': round(total_losses_kw, 2),
            'balance_error_percent': round(balance_error_percent, 4),
            'balance_closure_valid': balance_error_percent < 2.0,
            'calculation_method': 'first_law_direct',
            'standards': ['ASME_PTC_4', 'ISO_50001']
        }
```

### 2. Second Law (Exergy) Efficiency Calculator

**File**: `calculators/second_law_efficiency_calculator.py`

**Purpose**: Calculate exergy (available work) efficiency based on Second Law of Thermodynamics

**Formula**:
```
eta_2 = (Exergy Output) / (Exergy Input) x 100%

Where:
- Exergy = H - T0 x S (enthalpy minus ambient temperature times entropy)
- T0 = Reference (ambient) temperature (usually 298.15 K)
```

**Implementation**:
```python
class SecondLawEfficiencyCalculator:
    """
    Deterministic Second Law (Exergy) efficiency calculator.

    Physics Basis: Second Law of Thermodynamics (Exergy Analysis)
    Standards: ASME PTC 4.1, ISO 50001, Kotas Exergy Method
    """

    AMBIENT_TEMPERATURE_K = 298.15  # 25C reference

    def calculate(
        self,
        enthalpy_in_kj_kg: float,
        entropy_in_kj_kg_k: float,
        enthalpy_out_kj_kg: float,
        entropy_out_kj_kg_k: float,
        mass_flow_kg_s: float,
        ambient_temperature_k: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate Second Law (Exergy) efficiency.

        Args:
            enthalpy_in_kj_kg: Inlet enthalpy (kJ/kg)
            entropy_in_kj_kg_k: Inlet entropy (kJ/kg-K)
            enthalpy_out_kj_kg: Outlet enthalpy (kJ/kg)
            entropy_out_kj_kg_k: Outlet entropy (kJ/kg-K)
            mass_flow_kg_s: Mass flow rate (kg/s)
            ambient_temperature_k: Ambient temperature (K), default 298.15

        Returns:
            Exergy efficiency result with breakdown
        """
        T0 = ambient_temperature_k or self.AMBIENT_TEMPERATURE_K

        # Exergy calculations: Ex = H - T0 * S
        exergy_in_kj_kg = enthalpy_in_kj_kg - T0 * entropy_in_kj_kg_k
        exergy_out_kj_kg = enthalpy_out_kj_kg - T0 * entropy_out_kj_kg_k

        # Exergy flow rates
        exergy_in_kw = exergy_in_kj_kg * mass_flow_kg_s
        exergy_out_kw = exergy_out_kj_kg * mass_flow_kg_s

        # Exergy destruction (irreversibility)
        exergy_destruction_kw = exergy_in_kw - exergy_out_kw

        # Second Law efficiency
        if exergy_in_kw > 0:
            efficiency_percent = (exergy_out_kw / exergy_in_kw) * 100
        else:
            efficiency_percent = 0.0

        return {
            'exergy_efficiency_percent': round(efficiency_percent, 2),
            'exergy_input_kw': round(exergy_in_kw, 2),
            'exergy_output_kw': round(exergy_out_kw, 2),
            'exergy_destruction_kw': round(exergy_destruction_kw, 2),
            'irreversibility_percent': round(
                (exergy_destruction_kw / exergy_in_kw * 100) if exergy_in_kw > 0 else 0, 2
            ),
            'ambient_temperature_k': T0,
            'calculation_method': 'exergy_analysis',
            'standards': ['ASME_PTC_4.1', 'Kotas_Method']
        }
```

### 3. Heat Loss Calculator

**File**: `calculators/heat_loss_calculator.py`

**Purpose**: Calculate and categorize all heat losses (radiation, convection, conduction, flue gas, unburned fuel)

**Formulas**:
```
Q_loss = Q_radiation + Q_convection + Q_conduction + Q_flue + Q_unburned

Q_radiation = epsilon * sigma * A * (T_surface^4 - T_ambient^4)
Q_convection = h * A * (T_surface - T_ambient)
Q_conduction = k * A * (T_hot - T_cold) / L
Q_flue = m_flue * Cp_flue * (T_stack - T_ambient)
Q_unburned = m_unburned * HHV_fuel
```

**Implementation**:
```python
class HeatLossCalculator:
    """
    Comprehensive heat loss calculator for thermal systems.

    Physics Basis: Heat transfer fundamentals (Incropera & DeWitt)
    Standards: ASME PTC 4, DOE AMO Steam Tip Sheets
    """

    STEFAN_BOLTZMANN = 5.67e-8  # W/m^2-K^4

    def calculate_radiation_loss(
        self,
        surface_area_m2: float,
        surface_temp_c: float,
        ambient_temp_c: float,
        emissivity: float = 0.85
    ) -> Dict[str, float]:
        """Calculate radiation heat loss using Stefan-Boltzmann law."""
        T_surface_k = surface_temp_c + 273.15
        T_ambient_k = ambient_temp_c + 273.15

        Q_radiation_w = (
            emissivity * self.STEFAN_BOLTZMANN * surface_area_m2 *
            (T_surface_k**4 - T_ambient_k**4)
        )

        return {
            'radiation_loss_kw': round(Q_radiation_w / 1000, 2),
            'emissivity': emissivity,
            'surface_temp_k': round(T_surface_k, 2),
            'formula': 'stefan_boltzmann'
        }

    def calculate_convection_loss(
        self,
        surface_area_m2: float,
        surface_temp_c: float,
        ambient_temp_c: float,
        heat_transfer_coeff_w_m2k: float = 10.0  # Natural convection default
    ) -> Dict[str, float]:
        """Calculate convection heat loss using Newton's law of cooling."""
        Q_convection_w = (
            heat_transfer_coeff_w_m2k * surface_area_m2 *
            (surface_temp_c - ambient_temp_c)
        )

        return {
            'convection_loss_kw': round(Q_convection_w / 1000, 2),
            'heat_transfer_coeff': heat_transfer_coeff_w_m2k,
            'formula': 'newton_cooling'
        }

    def calculate_flue_gas_loss(
        self,
        flue_gas_flow_kg_hr: float,
        flue_gas_temp_c: float,
        ambient_temp_c: float,
        flue_gas_cp_kj_kgk: float = 1.1  # Typical combustion products
    ) -> Dict[str, float]:
        """Calculate sensible heat loss in flue gas."""
        Q_flue_kw = (
            flue_gas_flow_kg_hr * flue_gas_cp_kj_kgk *
            (flue_gas_temp_c - ambient_temp_c) / 3600
        )

        return {
            'flue_gas_loss_kw': round(Q_flue_kw, 2),
            'stack_temperature_c': flue_gas_temp_c,
            'flue_gas_cp': flue_gas_cp_kj_kgk,
            'formula': 'sensible_heat'
        }

    def calculate_total_losses(
        self,
        radiation: Dict[str, float],
        convection: Dict[str, float],
        conduction: Optional[Dict[str, float]] = None,
        flue_gas: Optional[Dict[str, float]] = None,
        unburned_fuel: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Aggregate all heat losses with breakdown."""
        losses = {
            'radiation_kw': radiation.get('radiation_loss_kw', 0),
            'convection_kw': convection.get('convection_loss_kw', 0),
            'conduction_kw': conduction.get('conduction_loss_kw', 0) if conduction else 0,
            'flue_gas_kw': flue_gas.get('flue_gas_loss_kw', 0) if flue_gas else 0,
            'unburned_fuel_kw': unburned_fuel.get('unburned_loss_kw', 0) if unburned_fuel else 0
        }

        total_loss_kw = sum(losses.values())

        # Calculate percentages
        loss_breakdown = {}
        for loss_type, value in losses.items():
            if total_loss_kw > 0:
                loss_breakdown[loss_type.replace('_kw', '_percent')] = round(
                    value / total_loss_kw * 100, 1
                )

        return {
            'total_loss_kw': round(total_loss_kw, 2),
            'loss_breakdown_kw': losses,
            'loss_breakdown_percent': loss_breakdown,
            'standards': ['ASME_PTC_4', 'DOE_AMO']
        }
```

### 4. Sankey Diagram Generator

**File**: `calculators/sankey_generator.py`

**Purpose**: Generate interactive Sankey diagrams for energy flow visualization

**Implementation**:
```python
class SankeyGenerator:
    """
    Plotly-based Sankey diagram generator for energy balance visualization.

    Design: Interactive HTML/SVG output with energy balance validation.
    """

    def generate(
        self,
        energy_input: Dict[str, float],
        useful_output: Dict[str, float],
        losses: Dict[str, float],
        title: str = "Thermal Energy Balance"
    ) -> Dict[str, Any]:
        """
        Generate Sankey diagram data structure.

        Args:
            energy_input: Input energy flows (e.g., {'fuel': 1000, 'electrical': 50})
            useful_output: Useful output flows (e.g., {'steam': 800, 'hot_water': 50})
            losses: Loss flows (e.g., {'flue_gas': 100, 'radiation': 20})
            title: Diagram title

        Returns:
            Sankey diagram data for Plotly rendering
        """
        # Validate energy balance
        total_input = sum(energy_input.values())
        total_output = sum(useful_output.values()) + sum(losses.values())
        balance_error = abs(total_input - total_output) / total_input * 100

        if balance_error > 2.0:
            raise ValueError(
                f"Energy balance error {balance_error:.2f}% exceeds 2% tolerance"
            )

        # Build node list
        nodes = []
        node_colors = []

        # Input nodes (green)
        for name in energy_input.keys():
            nodes.append(name.replace('_', ' ').title())
            node_colors.append('#2ecc71')

        # Central process node (blue)
        nodes.append('Process')
        node_colors.append('#3498db')
        process_idx = len(nodes) - 1

        # Output nodes (orange for useful, red for losses)
        for name in useful_output.keys():
            nodes.append(name.replace('_', ' ').title())
            node_colors.append('#f39c12')

        for name in losses.keys():
            nodes.append(name.replace('_', ' ').title() + ' Loss')
            node_colors.append('#e74c3c')

        # Build links
        sources = []
        targets = []
        values = []
        link_colors = []

        # Input to process
        for i, (name, value) in enumerate(energy_input.items()):
            sources.append(i)
            targets.append(process_idx)
            values.append(value)
            link_colors.append('rgba(46, 204, 113, 0.5)')

        # Process to outputs
        output_idx = process_idx + 1
        for name, value in useful_output.items():
            sources.append(process_idx)
            targets.append(output_idx)
            values.append(value)
            link_colors.append('rgba(243, 156, 18, 0.5)')
            output_idx += 1

        # Process to losses
        for name, value in losses.items():
            sources.append(process_idx)
            targets.append(output_idx)
            values.append(value)
            link_colors.append('rgba(231, 76, 60, 0.5)')
            output_idx += 1

        return {
            'diagram_type': 'sankey',
            'title': title,
            'nodes': {
                'labels': nodes,
                'colors': node_colors
            },
            'links': {
                'sources': sources,
                'targets': targets,
                'values': values,
                'colors': link_colors
            },
            'energy_balance': {
                'total_input_kw': round(total_input, 2),
                'total_output_kw': round(sum(useful_output.values()), 2),
                'total_losses_kw': round(sum(losses.values()), 2),
                'efficiency_percent': round(
                    sum(useful_output.values()) / total_input * 100, 2
                ),
                'balance_error_percent': round(balance_error, 4)
            },
            'plotly_figure': self._create_plotly_figure(
                nodes, node_colors, sources, targets, values, link_colors, title
            )
        }

    def _create_plotly_figure(
        self,
        nodes, node_colors, sources, targets, values, link_colors, title
    ) -> Dict[str, Any]:
        """Create Plotly figure specification."""
        return {
            'data': [{
                'type': 'sankey',
                'node': {
                    'label': nodes,
                    'color': node_colors,
                    'pad': 15,
                    'thickness': 20
                },
                'link': {
                    'source': sources,
                    'target': targets,
                    'value': values,
                    'color': link_colors
                }
            }],
            'layout': {
                'title': title,
                'font': {'size': 12},
                'height': 600,
                'width': 1000
            }
        }
```

### 5. Benchmark Calculator

**File**: `calculators/benchmark_calculator.py`

**Purpose**: Compare efficiency against industry benchmarks and best practices

**Implementation**:
```python
class BenchmarkCalculator:
    """
    Industry benchmark comparison calculator.

    Data Sources: DOE AMO, EPA ENERGY STAR, IEA Industrial Efficiency
    """

    # Industry benchmark database (embedded for determinism)
    BENCHMARKS = {
        'boiler_steam': {
            'bottom_quartile': 70.0,
            'median': 80.0,
            'top_quartile': 85.0,
            'best_in_class': 92.0,
            'theoretical_max': 95.0,
            'unit': 'percent',
            'source': 'DOE_AMO_Steam_Best_Practices'
        },
        'furnace_process': {
            'bottom_quartile': 35.0,
            'median': 50.0,
            'top_quartile': 65.0,
            'best_in_class': 80.0,
            'theoretical_max': 85.0,
            'unit': 'percent',
            'source': 'DOE_AMO_Process_Heating'
        },
        'heat_exchanger': {
            'bottom_quartile': 60.0,
            'median': 75.0,
            'top_quartile': 85.0,
            'best_in_class': 95.0,
            'theoretical_max': 98.0,
            'unit': 'percent',
            'source': 'ASME_PTC_12'
        },
        'cogeneration_chp': {
            'bottom_quartile': 55.0,
            'median': 70.0,
            'top_quartile': 80.0,
            'best_in_class': 85.0,
            'theoretical_max': 90.0,
            'unit': 'percent',
            'source': 'EPA_CHP_Partnership'
        }
    }

    def calculate(
        self,
        efficiency_percent: float,
        equipment_type: str,
        custom_benchmark: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Compare efficiency against industry benchmarks.

        Args:
            efficiency_percent: Current efficiency (%)
            equipment_type: Type of equipment for benchmark lookup
            custom_benchmark: Optional custom benchmark values

        Returns:
            Benchmark comparison with percentile ranking
        """
        benchmark = custom_benchmark or self.BENCHMARKS.get(equipment_type)

        if not benchmark:
            raise ValueError(f"Unknown equipment type: {equipment_type}")

        # Determine percentile ranking
        if efficiency_percent >= benchmark['best_in_class']:
            percentile = 95
            ranking = 'Best in Class'
        elif efficiency_percent >= benchmark['top_quartile']:
            percentile = 75 + (efficiency_percent - benchmark['top_quartile']) / (
                benchmark['best_in_class'] - benchmark['top_quartile']
            ) * 20
            ranking = 'Top Quartile'
        elif efficiency_percent >= benchmark['median']:
            percentile = 50 + (efficiency_percent - benchmark['median']) / (
                benchmark['top_quartile'] - benchmark['median']
            ) * 25
            ranking = 'Above Average'
        elif efficiency_percent >= benchmark['bottom_quartile']:
            percentile = 25 + (efficiency_percent - benchmark['bottom_quartile']) / (
                benchmark['median'] - benchmark['bottom_quartile']
            ) * 25
            ranking = 'Below Average'
        else:
            percentile = efficiency_percent / benchmark['bottom_quartile'] * 25
            ranking = 'Bottom Quartile'

        # Calculate improvement potential
        improvement_potential = benchmark['best_in_class'] - efficiency_percent
        theoretical_gap = benchmark['theoretical_max'] - efficiency_percent

        return {
            'current_efficiency_percent': round(efficiency_percent, 2),
            'equipment_type': equipment_type,
            'percentile_ranking': round(percentile, 1),
            'ranking_category': ranking,
            'benchmark_values': {
                'bottom_quartile': benchmark['bottom_quartile'],
                'median': benchmark['median'],
                'top_quartile': benchmark['top_quartile'],
                'best_in_class': benchmark['best_in_class'],
                'theoretical_max': benchmark['theoretical_max']
            },
            'improvement_potential': {
                'to_best_in_class_percent': round(improvement_potential, 2),
                'to_theoretical_max_percent': round(theoretical_gap, 2)
            },
            'data_source': benchmark.get('source', 'Custom'),
            'standards': ['DOE_AMO', 'EPA_ENERGY_STAR', 'ISO_50001']
        }
```

### 6. Improvement Analyzer

**File**: `calculators/improvement_analyzer.py`

**Purpose**: Identify and prioritize efficiency improvement opportunities

### 7. Uncertainty Quantifier

**File**: `calculators/uncertainty_quantifier.py`

**Purpose**: Quantify measurement uncertainty using Monte Carlo simulation (deterministic seed)

**Implementation**:
```python
class UncertaintyQuantifier:
    """
    Measurement uncertainty quantification using Monte Carlo simulation.

    Standards: GUM (Guide to Uncertainty in Measurement), ISO/IEC Guide 98-3
    """

    def __init__(self, seed: int = 42):
        """Initialize with deterministic seed for reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def quantify_efficiency_uncertainty(
        self,
        efficiency_percent: float,
        input_uncertainties: Dict[str, float],
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty in efficiency calculation.

        Args:
            efficiency_percent: Calculated efficiency (%)
            input_uncertainties: Relative uncertainties for each input (%)
            n_simulations: Number of Monte Carlo samples

        Returns:
            Uncertainty bounds and confidence intervals
        """
        # Reset RNG for determinism
        self.rng = np.random.default_rng(self.seed)

        # Propagate uncertainties using Monte Carlo
        combined_uncertainty = np.sqrt(
            sum(u**2 for u in input_uncertainties.values())
        )

        # Generate Monte Carlo samples
        samples = self.rng.normal(
            efficiency_percent,
            efficiency_percent * combined_uncertainty / 100,
            n_simulations
        )

        # Calculate confidence intervals
        percentiles = {
            '95_lower': np.percentile(samples, 2.5),
            '95_upper': np.percentile(samples, 97.5),
            '90_lower': np.percentile(samples, 5),
            '90_upper': np.percentile(samples, 95),
            '68_lower': np.percentile(samples, 16),
            '68_upper': np.percentile(samples, 84)
        }

        return {
            'efficiency_percent': round(efficiency_percent, 2),
            'combined_uncertainty_percent': round(combined_uncertainty, 2),
            'expanded_uncertainty_k2': round(combined_uncertainty * 2, 2),
            'confidence_intervals': {
                '95_percent': {
                    'lower': round(percentiles['95_lower'], 2),
                    'upper': round(percentiles['95_upper'], 2)
                },
                '90_percent': {
                    'lower': round(percentiles['90_lower'], 2),
                    'upper': round(percentiles['90_upper'], 2)
                },
                '68_percent': {
                    'lower': round(percentiles['68_lower'], 2),
                    'upper': round(percentiles['68_upper'], 2)
                }
            },
            'input_uncertainties': input_uncertainties,
            'n_simulations': n_simulations,
            'seed': self.seed,
            'standards': ['GUM', 'ISO_IEC_Guide_98-3']
        }
```

### 8. Fuel Energy Calculator

**File**: `calculators/fuel_energy_calculator.py`

**Purpose**: Calculate fuel energy content using Higher/Lower Heating Values (HHV/LHV)

**Implementation**:
```python
class FuelEnergyCalculator:
    """
    Fuel energy content calculator using standard heating values.

    Data Sources: API (American Petroleum Institute), ASTM D240
    """

    # Heating values database (MJ/kg)
    HEATING_VALUES = {
        'natural_gas': {'hhv': 55.5, 'lhv': 50.0, 'density_kg_m3': 0.68},
        'propane': {'hhv': 50.3, 'lhv': 46.4, 'density_kg_m3': 1.88},
        'diesel': {'hhv': 45.6, 'lhv': 43.0, 'density_kg_m3': 850.0},
        'fuel_oil_2': {'hhv': 45.5, 'lhv': 42.5, 'density_kg_m3': 870.0},
        'fuel_oil_6': {'hhv': 42.5, 'lhv': 40.0, 'density_kg_m3': 990.0},
        'coal_bituminous': {'hhv': 32.5, 'lhv': 31.0, 'density_kg_m3': 1350.0},
        'coal_anthracite': {'hhv': 34.0, 'lhv': 32.5, 'density_kg_m3': 1500.0},
        'wood_chips': {'hhv': 19.0, 'lhv': 16.0, 'density_kg_m3': 350.0},
        'biomass_pellets': {'hhv': 20.0, 'lhv': 18.0, 'density_kg_m3': 650.0},
        'hydrogen': {'hhv': 141.8, 'lhv': 120.0, 'density_kg_m3': 0.089}
    }

    def calculate(
        self,
        fuel_type: str,
        mass_flow_kg_hr: float,
        use_hhv: bool = True,
        custom_heating_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate fuel energy input rate.

        Args:
            fuel_type: Type of fuel
            mass_flow_kg_hr: Fuel mass flow rate (kg/hr)
            use_hhv: Use HHV (True) or LHV (False)
            custom_heating_value: Override heating value (MJ/kg)

        Returns:
            Energy input calculations
        """
        fuel_data = self.HEATING_VALUES.get(fuel_type)
        if not fuel_data and not custom_heating_value:
            raise ValueError(f"Unknown fuel type: {fuel_type}")

        if custom_heating_value:
            heating_value_mj_kg = custom_heating_value
        else:
            heating_value_mj_kg = fuel_data['hhv'] if use_hhv else fuel_data['lhv']

        # Energy calculations
        energy_mj_hr = mass_flow_kg_hr * heating_value_mj_kg
        energy_kw = energy_mj_hr / 3.6  # MJ/hr to kW
        energy_mmbtu_hr = energy_mj_hr * 0.000947817  # MJ to MMBtu

        return {
            'fuel_type': fuel_type,
            'mass_flow_kg_hr': round(mass_flow_kg_hr, 2),
            'heating_value_mj_kg': round(heating_value_mj_kg, 2),
            'heating_value_type': 'HHV' if use_hhv else 'LHV',
            'energy_input_kw': round(energy_kw, 2),
            'energy_input_mj_hr': round(energy_mj_hr, 2),
            'energy_input_mmbtu_hr': round(energy_mmbtu_hr, 4),
            'data_source': 'API_ASTM_D240',
            'standards': ['API', 'ASTM_D240']
        }
```

### 9. Steam Energy Calculator

**File**: `calculators/steam_energy_calculator.py`

**Purpose**: Calculate steam enthalpy and energy using IAPWS-IF97 steam tables

### 10. Electrical Energy Calculator

**File**: `calculators/electrical_energy_calculator.py`

**Purpose**: Calculate motor/pump electrical energy and efficiency

---

## Integration Connectors

### 2.3 Integration Connectors Architecture

```
+===========================================================================+
|                      INTEGRATION CONNECTORS LAYER                         |
+===========================================================================+
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    ENERGY METER CONNECTOR                          |   |
|  |  Protocol: Modbus TCP/RTU, OPC-UA                                  |   |
|  |  Meters: Schneider, ABB, Siemens, Eaton, Landis+Gyr               |   |
|  |  Data: Power (kW), Energy (kWh), PF, Voltage, Current             |   |
|  |  Polling: 1-60 seconds configurable                               |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    HISTORIAN CONNECTOR                             |   |
|  |  Systems: OSIsoft PI, Wonderware (AVEVA), AspenTech IP.21         |   |
|  |  Protocol: Native SDK, OPC-HDA, REST API                          |   |
|  |  Data: Historical tag values, aggregations, interpolation         |   |
|  |  Query: Time-range, snapshot, calculated expressions              |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    SCADA CONNECTOR                                 |   |
|  |  Systems: Ignition, Wonderware InTouch, FactoryTalk, WinCC        |   |
|  |  Protocol: OPC-DA, OPC-UA, native drivers                         |   |
|  |  Data: Real-time process variables, setpoints, alarms             |   |
|  |  Features: Tag subscription, change-of-value, buffering           |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    ERP CONNECTOR                                   |   |
|  |  Systems: SAP S/4HANA, Oracle EBS, Microsoft Dynamics             |   |
|  |  Protocol: RFC/BAPI, REST API, OData                              |   |
|  |  Data: Energy costs, fuel prices, production schedules            |   |
|  |  Features: Cost allocation, invoice matching, budget tracking     |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```

### Energy Meter Connector

**File**: `connectors/energy_meter_connector.py`

```python
class EnergyMeterConnector:
    """
    Energy meter integration via Modbus and OPC-UA protocols.

    Supported Meters: Schneider PM5xxx, ABB B2x, Siemens SENTRON
    """

    def __init__(self, config: Dict[str, Any]):
        self.protocol = config.get('protocol', 'modbus_tcp')
        self.host = config['host']
        self.port = config.get('port', 502)
        self.slave_id = config.get('slave_id', 1)
        self.poll_interval_s = config.get('poll_interval_s', 5)

    async def read_power_kw(self) -> Dict[str, float]:
        """Read instantaneous power from meter."""
        if self.protocol == 'modbus_tcp':
            return await self._read_modbus_power()
        elif self.protocol == 'opc_ua':
            return await self._read_opcua_power()
        else:
            raise ValueError(f"Unknown protocol: {self.protocol}")

    async def read_energy_kwh(self) -> Dict[str, float]:
        """Read cumulative energy from meter."""
        # Implementation details...
        pass
```

### Historian Connector

**File**: `connectors/historian_connector.py`

```python
class HistorianConnector:
    """
    Process historian integration for OSIsoft PI, Wonderware, AspenTech.
    """

    SUPPORTED_SYSTEMS = ['osisoft_pi', 'wonderware', 'aspentech_ip21']

    def __init__(self, config: Dict[str, Any]):
        self.system = config['system']
        self.server = config['server']
        self.credentials = config.get('credentials')

        if self.system not in self.SUPPORTED_SYSTEMS:
            raise ValueError(f"Unsupported historian: {self.system}")

    async def query_tag_values(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query historical tag values from historian."""
        pass

    async def query_aggregates(
        self,
        tag_names: List[str],
        start_time: datetime,
        end_time: datetime,
        aggregate_type: str = 'average'  # average, min, max, sum
    ) -> Dict[str, float]:
        """Query aggregated values from historian."""
        pass
```

### SCADA Connector

**File**: `connectors/scada_connector.py`

```python
class SCADAConnector:
    """
    SCADA/DCS integration for real-time process data.

    Supported Systems: Ignition, Wonderware InTouch, FactoryTalk, WinCC
    """

    def __init__(self, config: Dict[str, Any]):
        self.system = config['system']
        self.opc_server = config.get('opc_server')
        self.subscription_interval_ms = config.get('subscription_interval_ms', 1000)
        self._subscriptions = {}

    async def subscribe_to_tags(
        self,
        tag_names: List[str],
        callback: Callable[[str, Any, datetime], None]
    ) -> str:
        """Subscribe to tag value changes."""
        pass

    async def read_current_values(
        self,
        tag_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Read current values for multiple tags."""
        pass
```

---

## Visualization Engine

### 2.4 Visualization Engine Architecture

```
+===========================================================================+
|                      VISUALIZATION ENGINE                                 |
+===========================================================================+
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    SANKEY DIAGRAM GENERATOR                        |   |
|  |  Library: Plotly.js                                                |   |
|  |  Features: Interactive, hover tooltips, export PNG/SVG            |   |
|  |  Validation: Energy balance closure check (<2%)                   |   |
|  |  Output: HTML, JSON (Plotly spec), PNG, SVG                       |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    WATERFALL CHART GENERATOR                       |   |
|  |  Library: Matplotlib / Plotly                                      |   |
|  |  Purpose: Heat balance breakdown (input -> losses -> output)      |   |
|  |  Features: Stacked losses, color-coded categories                 |   |
|  |  Output: PNG, SVG, PDF                                            |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    EFFICIENCY TREND VISUALIZER                     |   |
|  |  Library: Plotly / Matplotlib                                      |   |
|  |  Purpose: Time-series efficiency tracking                         |   |
|  |  Features: Moving average, baseline comparison, annotations       |   |
|  |  Output: Interactive HTML, PNG                                    |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |                    LOSS BREAKDOWN PIE CHART                        |   |
|  |  Library: Plotly / Matplotlib                                      |   |
|  |  Purpose: Proportional loss visualization                         |   |
|  |  Features: Exploded segments, percentage labels                   |   |
|  |  Output: PNG, SVG, PDF                                            |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```

---

## Data Flow Architecture

### Complete Data Flow Pipeline

```
+===========================================================================+
|                      DATA FLOW ARCHITECTURE                               |
+===========================================================================+
|                                                                           |
|  PHASE 1: DATA INGESTION                                                  |
|  +-------------------------------------------------------------------+   |
|  |                                                                    |   |
|  |  Energy Meters  ──┐                                                |   |
|  |  (Modbus/OPC-UA) │                                                |   |
|  |                   │     +------------------+                       |   |
|  |  Historians     ──┼────>│   Data Intake    │                       |   |
|  |  (PI, Wonderware)│     │   Layer          │                       |   |
|  |                   │     +--------+---------+                       |   |
|  |  SCADA/DCS      ──┤              │                                 |   |
|  |  (Real-time)     │              v                                 |   |
|  |                   │     +------------------+                       |   |
|  |  ERP Systems    ──┘     │   Validation &   │                       |   |
|  |  (Cost Data)            │   Normalization  │                       |   |
|  |                         +--------+---------+                       |   |
|  +-------------------------------------------------------------------+   |
|                                     │                                     |
|  PHASE 2: CALCULATION               v                                     |
|  +-------------------------------------------------------------------+   |
|  |                                                                    |   |
|  |                    +------------------+                            |   |
|  |                    │   Cache Check    │                            |   |
|  |                    │   (TTL: 60s)     │                            |   |
|  |                    +--------+---------+                            |   |
|  |                             │                                      |   |
|  |              Cache Hit      │      Cache Miss                      |   |
|  |               ┌─────────────┴─────────────┐                        |   |
|  |               v                           v                        |   |
|  |    +------------------+       +------------------+                 |   |
|  |    │   Return Cached  │       │   Calculation    │                 |   |
|  |    │   Result         │       │   Engine         │                 |   |
|  |    +------------------+       +--------+---------+                 |   |
|  |                                        │                           |   |
|  |                          ┌─────────────┼─────────────┐             |   |
|  |                          v             v             v             |   |
|  |                    +---------+   +---------+   +---------+         |   |
|  |                    │First Law│   │Second   │   │Heat Loss│         |   |
|  |                    │Calc     │   │Law Calc │   │Calc     │         |   |
|  |                    +---------+   +---------+   +---------+         |   |
|  |                          │             │             │             |   |
|  |                          └─────────────┼─────────────┘             |   |
|  |                                        v                           |   |
|  +-------------------------------------------------------------------+   |
|                                           │                               |
|  PHASE 3: ANALYSIS                        v                               |
|  +-------------------------------------------------------------------+   |
|  |                                                                    |   |
|  |              +------------------+                                  |   |
|  |              │   Aggregation    │                                  |   |
|  |              +--------+---------+                                  |   |
|  |                       │                                            |   |
|  |        ┌──────────────┼──────────────┐                             |   |
|  |        v              v              v                             |   |
|  |  +---------+    +---------+    +---------+                         |   |
|  |  │Benchmark│    │Improve  │    │Uncertain│                         |   |
|  |  │Analysis │    │Analyzer │    │Quantify │                         |   |
|  |  +---------+    +---------+    +---------+                         |   |
|  |        │              │              │                             |   |
|  |        └──────────────┼──────────────┘                             |   |
|  |                       v                                            |   |
|  +-------------------------------------------------------------------+   |
|                                           │                               |
|  PHASE 4: VISUALIZATION                   v                               |
|  +-------------------------------------------------------------------+   |
|  |                                                                    |   |
|  |              +------------------+                                  |   |
|  |              │   Sankey         │                                  |   |
|  |              │   Generator      │                                  |   |
|  |              +--------+---------+                                  |   |
|  |                       │                                            |   |
|  |        ┌──────────────┼──────────────┐                             |   |
|  |        v              v              v                             |   |
|  |  +---------+    +---------+    +---------+                         |   |
|  |  │Waterfall│    │Trend    │    │Pie Chart│                         |   |
|  |  │Chart    │    │Analysis │    │(Losses) │                         |   |
|  |  +---------+    +---------+    +---------+                         |   |
|  |        │              │              │                             |   |
|  |        └──────────────┼──────────────┘                             |   |
|  |                       v                                            |   |
|  +-------------------------------------------------------------------+   |
|                                           │                               |
|  PHASE 5: OUTPUT                          v                               |
|  +-------------------------------------------------------------------+   |
|  |                                                                    |   |
|  |        ┌──────────────┼──────────────┐                             |   |
|  |        v              v              v                             |   |
|  |  +---------+    +---------+    +---------+                         |   |
|  |  │JSON/API │    │PDF      │    │CSV      │                         |   |
|  |  │Response │    │Report   │    │Export   │                         |   |
|  |  +---------+    +---------+    +---------+                         |   |
|  |        │              │              │                             |   |
|  |        v              v              v                             |   |
|  |  +-------------------------------------------------+               |   |
|  |  │           Provenance Tracker                    │               |   |
|  |  │           (SHA-256 Hash Chain)                  │               |   |
|  |  +-------------------------------------------------+               |   |
|  |                       │                                            |   |
|  |                       v                                            |   |
|  |  +-------------------------------------------------+               |   |
|  |  │           Audit Log (7 Year Retention)          │               |   |
|  |  +-------------------------------------------------+               |   |
|  |                                                                    |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
+===========================================================================+
```

---

## Physics Formulas (Zero-Hallucination)

### 4. Physics Formulas Reference

All calculations in GL-009 THERMALIQ are based on validated physics equations. **NO LLM is used in any calculation path.**

### First Law Efficiency (Energy Conservation)

```
+===========================================================================+
|                    FIRST LAW EFFICIENCY                                   |
+===========================================================================+
|                                                                           |
|  Formula:                                                                 |
|                                                                           |
|        eta_1 = (Q_useful / Q_input) x 100%                               |
|                                                                           |
|  Where:                                                                   |
|    Q_useful = Useful heat output (kW or kJ/s)                            |
|    Q_input  = Total energy input (fuel + electrical) (kW or kJ/s)        |
|                                                                           |
|  Energy Balance:                                                          |
|                                                                           |
|        Q_input = Q_useful + Q_losses                                     |
|                                                                           |
|  Expanded:                                                                |
|                                                                           |
|        Q_input = Q_useful + Q_rad + Q_conv + Q_cond + Q_flue + Q_unburned|
|                                                                           |
|  Standards: ASME PTC 4, ISO 50001, DOE AMO                               |
|                                                                           |
+===========================================================================+
```

### Second Law (Exergy) Efficiency

```
+===========================================================================+
|                    SECOND LAW (EXERGY) EFFICIENCY                         |
+===========================================================================+
|                                                                           |
|  Formula:                                                                 |
|                                                                           |
|        eta_2 = (Exergy_out / Exergy_in) x 100%                           |
|                                                                           |
|  Exergy Definition:                                                       |
|                                                                           |
|        Exergy = H - T_0 x S                                              |
|                                                                           |
|  Where:                                                                   |
|    H   = Enthalpy (kJ/kg)                                                |
|    S   = Entropy (kJ/kg-K)                                               |
|    T_0 = Reference (ambient) temperature (K), typically 298.15 K         |
|                                                                           |
|  Exergy Destruction (Irreversibility):                                    |
|                                                                           |
|        I = Exergy_in - Exergy_out                                        |
|                                                                           |
|  Irreversibility Rate:                                                    |
|                                                                           |
|        I_rate = (I / Exergy_in) x 100%                                   |
|                                                                           |
|  Standards: ASME PTC 4.1, Kotas Exergy Method                            |
|                                                                           |
+===========================================================================+
```

### Heat Loss Formulas

```
+===========================================================================+
|                    HEAT LOSS CALCULATIONS                                 |
+===========================================================================+
|                                                                           |
|  TOTAL HEAT LOSS:                                                         |
|                                                                           |
|    Q_loss = Q_radiation + Q_convection + Q_conduction + Q_flue + Q_unburned
|                                                                           |
|  ---------------------------------------------------------------------------
|                                                                           |
|  RADIATION LOSS (Stefan-Boltzmann Law):                                   |
|                                                                           |
|    Q_rad = epsilon x sigma x A x (T_surface^4 - T_ambient^4)             |
|                                                                           |
|  Where:                                                                   |
|    epsilon = Surface emissivity (0.1-1.0, typically 0.85 for oxidized metal)
|    sigma   = Stefan-Boltzmann constant (5.67 x 10^-8 W/m^2-K^4)          |
|    A       = Surface area (m^2)                                          |
|    T       = Absolute temperature (K)                                    |
|                                                                           |
|  ---------------------------------------------------------------------------
|                                                                           |
|  CONVECTION LOSS (Newton's Law of Cooling):                               |
|                                                                           |
|    Q_conv = h x A x (T_surface - T_ambient)                              |
|                                                                           |
|  Where:                                                                   |
|    h = Heat transfer coefficient (W/m^2-K)                               |
|        - Natural convection: 5-25 W/m^2-K                                |
|        - Forced convection: 25-250 W/m^2-K                               |
|    A = Surface area (m^2)                                                |
|    T = Temperature (C or K)                                              |
|                                                                           |
|  ---------------------------------------------------------------------------
|                                                                           |
|  CONDUCTION LOSS (Fourier's Law):                                         |
|                                                                           |
|    Q_cond = k x A x (T_hot - T_cold) / L                                 |
|                                                                           |
|  Where:                                                                   |
|    k = Thermal conductivity (W/m-K)                                      |
|    A = Cross-sectional area (m^2)                                        |
|    L = Thickness (m)                                                     |
|    T = Temperature (C or K)                                              |
|                                                                           |
|  ---------------------------------------------------------------------------
|                                                                           |
|  FLUE GAS LOSS (Sensible Heat):                                           |
|                                                                           |
|    Q_flue = m_flue x Cp_flue x (T_stack - T_ambient)                     |
|                                                                           |
|  Where:                                                                   |
|    m_flue   = Flue gas mass flow rate (kg/s)                             |
|    Cp_flue  = Specific heat of flue gas (~1.1 kJ/kg-K)                   |
|    T_stack  = Stack temperature (C)                                      |
|    T_ambient = Ambient temperature (C)                                    |
|                                                                           |
|  ---------------------------------------------------------------------------
|                                                                           |
|  UNBURNED FUEL LOSS:                                                      |
|                                                                           |
|    Q_unburned = m_unburned x HHV_fuel                                    |
|                                                                           |
|  Where:                                                                   |
|    m_unburned = Mass flow of unburned fuel (kg/s)                        |
|    HHV_fuel   = Higher Heating Value of fuel (kJ/kg)                     |
|                                                                           |
|  Standards: ASME PTC 4, Incropera & DeWitt Heat Transfer                 |
|                                                                           |
+===========================================================================+
```

### Sankey Energy Balance Validation

```
+===========================================================================+
|                    SANKEY ENERGY BALANCE                                  |
+===========================================================================+
|                                                                           |
|  Conservation Requirement:                                                |
|                                                                           |
|        SUM(Q_in) = SUM(Q_useful) + SUM(Q_losses)                         |
|                                                                           |
|  Balance Closure Check:                                                   |
|                                                                           |
|        error = |SUM(Q_in) - SUM(Q_useful) - SUM(Q_losses)| / SUM(Q_in) x 100%
|                                                                           |
|  Acceptance Criterion:                                                    |
|                                                                           |
|        error < 2.0%   (PASS)                                             |
|        error >= 2.0%  (FAIL - Recalibrate instruments)                   |
|                                                                           |
|  Visualization:                                                           |
|                                                                           |
|    +--------+                                                             |
|    | Fuel   |----+                                                        |
|    | 1000kW |    |    +--------+                                         |
|    +--------+    +--->| Process|----> Steam Output (800 kW)               |
|                       |        |----> Hot Water (50 kW)                   |
|    +--------+    +--->|        |----> Flue Gas Loss (100 kW)              |
|    | Elec   |----+    |        |----> Radiation Loss (30 kW)              |
|    | 50kW   |         +--------+----> Convection Loss (20 kW)             |
|    +--------+                                                             |
|                                                                           |
|    Input: 1050 kW   |   Output: 850 kW   |   Losses: 150 kW              |
|    Balance: 1050 = 850 + 150 + 50 (auxiliary) = 1050  (CLOSED)            |
|                                                                           |
+===========================================================================+
```

---

## Tool Specifications

### 5. Tool Specifications (JSON Schemas)

### Tool 1: calculate_first_law_efficiency

```json
{
  "name": "calculate_first_law_efficiency",
  "description": "Calculate First Law thermal efficiency based on energy conservation",
  "deterministic": true,
  "standards": ["ASME_PTC_4", "ISO_50001"],
  "input_schema": {
    "type": "object",
    "properties": {
      "energy_input_kw": {
        "type": "number",
        "minimum": 0.1,
        "maximum": 1000000,
        "description": "Total energy input (kW)"
      },
      "useful_output_kw": {
        "type": "number",
        "minimum": 0,
        "maximum": 1000000,
        "description": "Useful energy output (kW)"
      },
      "losses_breakdown": {
        "type": "object",
        "properties": {
          "radiation_kw": {"type": "number", "minimum": 0},
          "convection_kw": {"type": "number", "minimum": 0},
          "conduction_kw": {"type": "number", "minimum": 0},
          "flue_gas_kw": {"type": "number", "minimum": 0},
          "unburned_fuel_kw": {"type": "number", "minimum": 0}
        },
        "description": "Optional breakdown of heat losses"
      }
    },
    "required": ["energy_input_kw", "useful_output_kw"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "efficiency_percent": {"type": "number"},
      "energy_input_kw": {"type": "number"},
      "useful_output_kw": {"type": "number"},
      "total_losses_kw": {"type": "number"},
      "balance_error_percent": {"type": "number"},
      "balance_closure_valid": {"type": "boolean"}
    }
  }
}
```

### Tool 2: calculate_second_law_efficiency

```json
{
  "name": "calculate_second_law_efficiency",
  "description": "Calculate Second Law (Exergy) efficiency based on available work",
  "deterministic": true,
  "standards": ["ASME_PTC_4.1", "Kotas_Method"],
  "input_schema": {
    "type": "object",
    "properties": {
      "enthalpy_in_kj_kg": {
        "type": "number",
        "description": "Inlet stream enthalpy (kJ/kg)"
      },
      "entropy_in_kj_kg_k": {
        "type": "number",
        "description": "Inlet stream entropy (kJ/kg-K)"
      },
      "enthalpy_out_kj_kg": {
        "type": "number",
        "description": "Outlet stream enthalpy (kJ/kg)"
      },
      "entropy_out_kj_kg_k": {
        "type": "number",
        "description": "Outlet stream entropy (kJ/kg-K)"
      },
      "mass_flow_kg_s": {
        "type": "number",
        "minimum": 0,
        "description": "Mass flow rate (kg/s)"
      },
      "ambient_temperature_k": {
        "type": "number",
        "default": 298.15,
        "description": "Reference temperature (K)"
      }
    },
    "required": ["enthalpy_in_kj_kg", "entropy_in_kj_kg_k",
                 "enthalpy_out_kj_kg", "entropy_out_kj_kg_k", "mass_flow_kg_s"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "exergy_efficiency_percent": {"type": "number"},
      "exergy_input_kw": {"type": "number"},
      "exergy_output_kw": {"type": "number"},
      "exergy_destruction_kw": {"type": "number"},
      "irreversibility_percent": {"type": "number"}
    }
  }
}
```

### Tool 3: calculate_heat_losses

```json
{
  "name": "calculate_heat_losses",
  "description": "Calculate comprehensive heat loss breakdown",
  "deterministic": true,
  "standards": ["ASME_PTC_4", "DOE_AMO"],
  "input_schema": {
    "type": "object",
    "properties": {
      "surface_area_m2": {"type": "number", "minimum": 0.1},
      "surface_temp_c": {"type": "number", "minimum": -50, "maximum": 1500},
      "ambient_temp_c": {"type": "number", "minimum": -50, "maximum": 60},
      "emissivity": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.85},
      "heat_transfer_coeff_w_m2k": {"type": "number", "default": 10.0},
      "flue_gas_flow_kg_hr": {"type": "number", "minimum": 0},
      "flue_gas_temp_c": {"type": "number"},
      "flue_gas_cp_kj_kgk": {"type": "number", "default": 1.1}
    },
    "required": ["surface_area_m2", "surface_temp_c", "ambient_temp_c"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "total_loss_kw": {"type": "number"},
      "radiation_loss_kw": {"type": "number"},
      "convection_loss_kw": {"type": "number"},
      "flue_gas_loss_kw": {"type": "number"},
      "loss_breakdown_percent": {"type": "object"}
    }
  }
}
```

### Tool 4: generate_sankey_diagram

```json
{
  "name": "generate_sankey_diagram",
  "description": "Generate interactive Sankey energy flow diagram",
  "deterministic": true,
  "input_schema": {
    "type": "object",
    "properties": {
      "energy_inputs": {
        "type": "object",
        "additionalProperties": {"type": "number"},
        "description": "Input energy flows (name: value_kw)"
      },
      "useful_outputs": {
        "type": "object",
        "additionalProperties": {"type": "number"},
        "description": "Useful output flows (name: value_kw)"
      },
      "losses": {
        "type": "object",
        "additionalProperties": {"type": "number"},
        "description": "Loss flows (name: value_kw)"
      },
      "title": {"type": "string", "default": "Energy Balance"},
      "output_format": {
        "type": "string",
        "enum": ["json", "html", "png", "svg"],
        "default": "json"
      }
    },
    "required": ["energy_inputs", "useful_outputs", "losses"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "diagram_type": {"type": "string"},
      "plotly_figure": {"type": "object"},
      "energy_balance": {
        "type": "object",
        "properties": {
          "total_input_kw": {"type": "number"},
          "total_output_kw": {"type": "number"},
          "total_losses_kw": {"type": "number"},
          "efficiency_percent": {"type": "number"},
          "balance_error_percent": {"type": "number"}
        }
      }
    }
  }
}
```

### Tool 5: benchmark_efficiency

```json
{
  "name": "benchmark_efficiency",
  "description": "Compare efficiency against industry benchmarks",
  "deterministic": true,
  "input_schema": {
    "type": "object",
    "properties": {
      "efficiency_percent": {"type": "number", "minimum": 0, "maximum": 100},
      "equipment_type": {
        "type": "string",
        "enum": ["boiler_steam", "furnace_process", "heat_exchanger", "cogeneration_chp"]
      },
      "custom_benchmark": {
        "type": "object",
        "description": "Optional custom benchmark values"
      }
    },
    "required": ["efficiency_percent", "equipment_type"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "percentile_ranking": {"type": "number"},
      "ranking_category": {"type": "string"},
      "improvement_potential": {"type": "object"}
    }
  }
}
```

### Tool 6-10: Additional Tools

Similar JSON schemas defined for:
- `analyze_improvement_opportunities`
- `quantify_uncertainty`
- `calculate_fuel_energy`
- `calculate_steam_energy`
- `calculate_electrical_efficiency`

---

## Performance Architecture

### 6. Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Calculation Latency** | <500ms | P95 response time |
| **Sankey Generation** | <2s | P95 render time |
| **Memory Usage** | <1GB | Peak RSS during operation |
| **Cache Hit Rate** | >85% | Cache hits / total requests |
| **Throughput** | >100 calc/min | Sustained calculation rate |
| **API Response** | <200ms (cached) | P50 response time |
| **Energy Balance Closure** | <2% error | Validation check |

### Performance Optimization Strategies

```
+===========================================================================+
|                    PERFORMANCE ARCHITECTURE                               |
+===========================================================================+
|                                                                           |
|  1. THREAD-SAFE CACHING                                                   |
|  +-----------------------------------------------------------------+     |
|  |  - LRU cache with TTL (60 seconds default)                      |     |
|  |  - Max size: 500 entries                                        |     |
|  |  - Cache key: SHA-256(input_data)                               |     |
|  |  - Target: 85%+ hit rate                                        |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
|  2. ASYNC EXECUTION                                                       |
|  +-----------------------------------------------------------------+     |
|  |  - asyncio for I/O-bound operations                             |     |
|  |  - ThreadPoolExecutor for CPU-bound calculations                |     |
|  |  - Parallel connector queries                                   |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
|  3. CONNECTION POOLING                                                    |
|  +-----------------------------------------------------------------+     |
|  |  - Database: SQLAlchemy async pool (20 connections)             |     |
|  |  - OPC-UA: Persistent subscriptions                             |     |
|  |  - Redis: aioredis connection pool                              |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
|  4. LAZY LOADING                                                          |
|  +-----------------------------------------------------------------+     |
|  |  - Steam tables loaded on first use                             |     |
|  |  - ML models loaded on demand                                   |     |
|  |  - Visualization libraries deferred import                      |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
+===========================================================================+
```

---

## Security Architecture

### 7. Security Architecture

```
+===========================================================================+
|                    SECURITY ARCHITECTURE                                  |
+===========================================================================+
|                                                                           |
|  LAYER 1: ZERO SECRETS POLICY                                             |
|  +-----------------------------------------------------------------+     |
|  |  - NO hardcoded credentials                                     |     |
|  |  - Environment variables for configuration                      |     |
|  |  - HashiCorp Vault integration for secrets                      |     |
|  |  - Kubernetes Secrets for container deployments                 |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
|  LAYER 2: JWT AUTHENTICATION                                              |
|  +-----------------------------------------------------------------+     |
|  |  - RS256 signed tokens                                          |     |
|  |  - Token expiration: 1 hour (configurable)                      |     |
|  |  - Refresh token rotation                                       |     |
|  |  - Scope-based authorization                                    |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
|  LAYER 3: AUDIT LOGGING                                                   |
|  +-----------------------------------------------------------------+     |
|  |  - All API calls logged with user context                       |     |
|  |  - Calculation inputs/outputs logged                            |     |
|  |  - Retention: 7 years (regulatory compliance)                   |     |
|  |  - Tamper-proof with blockchain-style linking                   |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
|  LAYER 4: PROVENANCE HASHING                                              |
|  +-----------------------------------------------------------------+     |
|  |  - SHA-256 hash of inputs + outputs                             |     |
|  |  - Hash chain for audit trail                                   |     |
|  |  - Deterministic - same inputs = same hash                      |     |
|  |  - Regulatory compliance (ISO 14064, GHG Protocol)              |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
|  LAYER 5: NETWORK SECURITY                                                |
|  +-----------------------------------------------------------------+     |
|  |  - TLS 1.3 for all communications                               |     |
|  |  - Network egress allowlist only                                |     |
|  |  - Kubernetes NetworkPolicy enforcement                         |     |
|  |  - API rate limiting (100 req/min default)                      |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
|  LAYER 6: RBAC (Role-Based Access Control)                                |
|  +-----------------------------------------------------------------+     |
|  |  Roles:                                                         |     |
|  |    - viewer: Read efficiency reports                            |     |
|  |    - operator: Run calculations, generate diagrams              |     |
|  |    - analyst: Benchmark analysis, improvement recommendations   |     |
|  |    - admin: Configure agent, manage integrations                |     |
|  +-----------------------------------------------------------------+     |
|                                                                           |
+===========================================================================+
```

---

## Deployment Architecture

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-009-thermaliq
  namespace: greenlang-agents
  labels:
    app: thermaliq
    agent-id: gl-009
    version: "1.0.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: thermaliq
  template:
    metadata:
      labels:
        app: thermaliq
        agent-id: gl-009
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: thermaliq
        image: greenlang/gl-009-thermaliq:1.0.0
        ports:
        - containerPort: 8080
          name: http
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: CACHE_TTL_SECONDS
          value: "60"
        - name: ENABLE_METRICS
          value: "true"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: gl-009-config
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-009-thermaliq-hpa
  namespace: greenlang-agents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-009-thermaliq
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Technology Stack

### Core Runtime

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Language | Python | 3.11+ | Core runtime |
| Framework | FastAPI | 0.104.0+ | Async REST API |
| Server | Uvicorn | 0.24.0+ | ASGI server |
| Validation | Pydantic | 2.5.0+ | Data validation |

### Data Processing

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Numerical | NumPy | 1.24.0+ | Array operations |
| DataFrames | Pandas | 2.1.0+ | Data manipulation |
| Steam Tables | iapws | 1.5.0+ | IAPWS-IF97 |
| Scientific | SciPy | 1.11.0+ | Scientific computing |

### Visualization

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Sankey | Plotly | 5.18.0+ | Interactive diagrams |
| Charts | Matplotlib | 3.8.0+ | Static charts |
| Reports | ReportLab | 4.0.0+ | PDF generation |

### Integration

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| OPC-UA | asyncua | 1.0.0+ | OPC-UA client |
| Modbus | pymodbus | 3.5.0+ | Modbus TCP/RTU |
| Database | asyncpg | 0.29.0+ | PostgreSQL async |
| Cache | aioredis | 2.0.0+ | Redis async |

### Security

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| JWT | python-jose | 3.3.0+ | Token handling |
| Crypto | cryptography | 41.0.0+ | Encryption |
| Hashing | hashlib | stdlib | SHA-256 |

### Deployment

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Container | Docker | 24.0+ | Containerization |
| Orchestration | Kubernetes | 1.28+ | Container orchestration |
| IaC | Terraform | 1.6+ | Infrastructure as code |
| CI/CD | GitHub Actions | - | Automation |

---

## Design Patterns

### 1. Strategy Pattern (Calculator Selection)

```python
class EfficiencyCalculator(ABC):
    @abstractmethod
    def calculate(self, data: Dict) -> Dict: pass

class FirstLawCalculator(EfficiencyCalculator):
    def calculate(self, data: Dict) -> Dict:
        # First Law implementation
        pass

class SecondLawCalculator(EfficiencyCalculator):
    def calculate(self, data: Dict) -> Dict:
        # Second Law implementation
        pass

# Usage
calculator = get_calculator(calculation_type)
result = calculator.calculate(energy_data)
```

### 2. Factory Pattern (Connector Creation)

```python
class ConnectorFactory:
    @staticmethod
    def create(connector_type: str, config: Dict) -> BaseConnector:
        if connector_type == 'modbus':
            return ModbusConnector(config)
        elif connector_type == 'opc_ua':
            return OPCUAConnector(config)
        elif connector_type == 'historian':
            return HistorianConnector(config)
        else:
            raise ValueError(f"Unknown connector: {connector_type}")
```

### 3. Observer Pattern (Real-Time Updates)

```python
class EfficiencyMonitor:
    def __init__(self):
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, efficiency_data):
        for observer in self.observers:
            observer.update(efficiency_data)

# Usage
monitor.attach(DashboardUpdater())
monitor.attach(AlertingService())
monitor.attach(DatabaseLogger())
```

### 4. Chain of Responsibility (Validation)

```python
class ValidationChain:
    def __init__(self):
        self.validators = [
            SchemaValidator(),
            RangeValidator(),
            UnitValidator(),
            EnergyBalanceValidator()
        ]

    def validate(self, data: Dict) -> ValidationResult:
        for validator in self.validators:
            result = validator.validate(data)
            if not result.is_valid:
                return result
        return ValidationResult(is_valid=True)
```

---

## Error Handling Strategy

### Error Categories

| Category | HTTP Status | Retry | Action |
|----------|-------------|-------|--------|
| Validation Error | 400 | No | Return field-level errors |
| Authentication Error | 401 | No | Require re-authentication |
| Authorization Error | 403 | No | Check permissions |
| Not Found | 404 | No | Return resource not found |
| Rate Limit | 429 | Yes | Wait and retry |
| Internal Error | 500 | Yes | Log and alert |
| Service Unavailable | 503 | Yes | Circuit breaker |

### Retry Policy

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(TransientError)
)
async def call_external_service():
    # External service call
    pass
```

---

## Monitoring & Observability

### Prometheus Metrics

```python
# Counters
efficiency_calculations_total = Counter(
    'thermaliq_calculations_total',
    'Total efficiency calculations',
    ['calculation_type', 'status']
)

# Histograms
calculation_duration_seconds = Histogram(
    'thermaliq_calculation_duration_seconds',
    'Calculation processing time',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# Gauges
current_efficiency_percent = Gauge(
    'thermaliq_efficiency_percent',
    'Current thermal efficiency',
    ['equipment_id', 'calculation_type']
)

cache_hit_ratio = Gauge(
    'thermaliq_cache_hit_ratio',
    'Cache hit ratio'
)
```

### Grafana Dashboards

1. **Efficiency Overview**: Real-time efficiency tracking, trends, benchmarks
2. **Performance**: API latency, throughput, error rates
3. **Heat Balance**: Sankey visualization, loss breakdown
4. **Alerts**: Critical efficiency drops, balance errors

---

## Future Enhancements

### Planned Features (v1.1)

1. **Digital Twin Integration**: Real-time simulation of thermal systems
2. **ML-Assisted Classification**: Equipment type auto-detection (non-calculation)
3. **Mobile App**: Field technician efficiency assessment tool
4. **Edge Deployment**: On-premises lightweight version
5. **Multi-Language Reports**: Localized PDF reports

### Research Areas

1. **Predictive Efficiency**: Forecast efficiency degradation
2. **Optimization AI**: Recommend setpoint changes (advisory only)
3. **Federated Learning**: Cross-facility efficiency learning
4. **AR Visualization**: Augmented reality heat loss visualization

---

## Appendices

### Appendix A: Industry Benchmark Data Sources

| Source | Data Type | Update Frequency |
|--------|-----------|------------------|
| DOE AMO | Best practices, target efficiencies | Annual |
| EPA ENERGY STAR | Industrial benchmarks | Annual |
| IEA Industrial Efficiency | Global benchmarks | Biennial |
| ASME PTC Standards | Test procedures | As published |

### Appendix B: Heating Value Reference

| Fuel | HHV (MJ/kg) | LHV (MJ/kg) | Source |
|------|-------------|-------------|--------|
| Natural Gas | 55.5 | 50.0 | API |
| Propane | 50.3 | 46.4 | API |
| Diesel | 45.6 | 43.0 | ASTM D240 |
| Coal (Bituminous) | 32.5 | 31.0 | ASTM D5865 |
| Wood Chips | 19.0 | 16.0 | DOE Biomass |
| Hydrogen | 141.8 | 120.0 | NIST |

### Appendix C: Heat Transfer Coefficients

| Condition | h (W/m^2-K) | Application |
|-----------|-------------|-------------|
| Natural Convection (Air) | 5-25 | Outdoor equipment |
| Forced Convection (Air) | 25-250 | Ventilated enclosures |
| Natural Convection (Water) | 200-1000 | Storage tanks |
| Forced Convection (Water) | 1000-15000 | Heat exchangers |
| Boiling | 2500-25000 | Evaporators |
| Condensation | 5000-25000 | Condensers |

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-26
**Authors**: GreenLang Foundation Agent Engineering Team
**Classification**: Internal - Production Documentation
**License**: Apache-2.0

---

For questions or clarifications, contact: agents@greenlang.org
