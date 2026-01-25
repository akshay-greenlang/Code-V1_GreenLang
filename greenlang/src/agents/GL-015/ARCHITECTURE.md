# GL-015 INSULSCAN Architecture Documentation

## Document Information

| Attribute | Value |
|-----------|-------|
| **Document Title** | GL-015 INSULSCAN System Architecture |
| **Agent ID** | GL-015 |
| **Codename** | INSULSCAN |
| **Version** | 1.0.0 |
| **Last Updated** | 2025-12-01 |
| **Status** | Production Ready |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Principles](#design-principles)
3. [High-Level Architecture](#high-level-architecture)
4. [Calculator Layer Design](#calculator-layer-design)
5. [Integration Layer Design](#integration-layer-design)
6. [Data Flow Architecture](#data-flow-architecture)
7. [Security Architecture](#security-architecture)
8. [Scalability Considerations](#scalability-considerations)
9. [Deployment Architecture](#deployment-architecture)
10. [Database Schema](#database-schema)
11. [API Architecture](#api-architecture)
12. [Error Handling](#error-handling)
13. [Performance Architecture](#performance-architecture)
14. [Observability Architecture](#observability-architecture)
15. [Disaster Recovery](#disaster-recovery)

---

## Architecture Overview

### System Context

GL-015 INSULSCAN operates within the GreenLang Agent ecosystem, providing industrial insulation inspection capabilities through thermal image analysis and heat loss quantification.

```
+==============================================================================+
|                           SYSTEM CONTEXT DIAGRAM                              |
+==============================================================================+
|                                                                               |
|   +---------------+     +---------------+     +------------------+            |
|   | IR Camera     |     | CMMS Systems  |     | GreenLang        |            |
|   | Systems       |     |               |     | Agents           |            |
|   |               |     | - SAP PM      |     |                  |            |
|   | - FLIR        |     | - IBM Maximo  |     | - GL-001         |            |
|   | - Fluke       |     | - Oracle EAM  |     |   THERMOSYNC     |            |
|   | - Testo       |     |               |     | - GL-006         |            |
|   | - Optris      |     |               |     |   HEATRECLAIM    |            |
|   +-------+-------+     +-------+-------+     +--------+---------+            |
|           |                     |                      |                      |
|           |                     |                      |                      |
|           v                     v                      v                      |
|   +-----------------------------------------------------------------------+   |
|   |                                                                       |   |
|   |                     GL-015 INSULSCAN AGENT                           |   |
|   |                                                                       |   |
|   |   +-------------------+    +-------------------+   +---------------+  |   |
|   |   | Thermal Image     |    | Heat Loss         |   | Repair        |  |   |
|   |   | Analysis          |    | Quantification    |   | Prioritization|  |   |
|   |   +-------------------+    +-------------------+   +---------------+  |   |
|   |                                                                       |   |
|   +-----------------------------------------------------------------------+   |
|           |                     |                      |                      |
|           v                     v                      v                      |
|   +---------------+     +---------------+     +------------------+            |
|   | Inspection    |     | Work Orders   |     | Energy Reports   |            |
|   | Reports       |     |               |     |                  |            |
|   +---------------+     +---------------+     +------------------+            |
|                                                                               |
+==============================================================================+
```

### Key Architectural Goals

| Goal | Description | Priority |
|------|-------------|----------|
| **Zero Hallucination** | All numeric outputs from deterministic calculations | Critical |
| **Auditability** | Complete provenance tracking for regulatory compliance | Critical |
| **Scalability** | Handle facility-wide inspections with 1000+ images | High |
| **Integration** | Seamless connectivity with cameras and CMMS | High |
| **Performance** | Sub-second response for individual calculations | Medium |
| **Extensibility** | Support new camera types and integrations | Medium |

---

## Design Principles

### 1. Separation of Concerns

The architecture strictly separates AI-assisted tasks from deterministic calculations:

```
+==============================================================================+
|                    SEPARATION OF CONCERNS                                     |
+==============================================================================+
|                                                                               |
|   +-------------------------------------+  +--------------------------------+ |
|   |     AI-ALLOWED ZONE                |  |    DETERMINISTIC ZONE          | |
|   |     (Pattern Recognition)          |  |    (Engineering Calculations)  | |
|   +-------------------------------------+  +--------------------------------+ |
|   |                                     |  |                                | |
|   | [x] Anomaly pattern classification |  | [x] Heat loss (W, kW)          | |
|   | [x] Damage type identification     |  | [x] Temperature (C, K)         | |
|   | [x] Report narrative generation    |  | [x] ROI, NPV, IRR              | |
|   | [x] Entity resolution              |  | [x] Surface temperature        | |
|   | [x] Work order descriptions        |  | [x] CUI risk scores            | |
|   |                                     |  | [x] Energy costs ($)           | |
|   | [ ] Numeric calculations           |  | [x] CO2 emissions (kg)         | |
|   | [ ] Temperature values             |  | [x] Payback periods            | |
|   | [ ] Heat flux values               |  | [x] Priority rankings          | |
|   | [ ] Cost estimates                 |  |                                | |
|   |                                     |  | Formula-driven outputs only    | |
|   +-------------------------------------+  +--------------------------------+ |
|                                                                               |
+==============================================================================+
```

### 2. Immutability and Pure Functions

Calculator modules use immutable data structures and pure functions:

```python
# CORRECT: Immutable result with frozen dataclass
@dataclass(frozen=True)
class HeatLossResult:
    total_heat_loss_w: Decimal
    surface_temperature_c: Decimal
    provenance_hash: str
    calculation_steps: Tuple[CalculationStep, ...]

# CORRECT: Pure function - no side effects
def calculate_conduction(
    delta_t: Decimal,
    thermal_resistance: Decimal,
    area: Decimal
) -> Decimal:
    return delta_t / thermal_resistance * area

# INCORRECT: Mutable state (not used)
class MutableCalculator:
    def __init__(self):
        self.last_result = None  # BAD: mutable state
```

### 3. Decimal Precision for Financial Calculations

All monetary and engineering calculations use Python Decimal:

```python
from decimal import Decimal, ROUND_HALF_UP

# Configuration
DEFAULT_PRECISION = 6
MONETARY_PRECISION = 2

# Heat loss calculation
heat_loss = Decimal("2450.123456")
rounded = heat_loss.quantize(
    Decimal("0.01"),
    rounding=ROUND_HALF_UP
)  # 2450.12

# Monetary calculation
annual_cost = Decimal("19600.00") * Decimal("0.12")
rounded_cost = annual_cost.quantize(
    Decimal("0.01"),
    rounding=ROUND_HALF_UP
)  # 2352.00
```

### 4. Provenance Tracking

Every calculation includes complete provenance for audit:

```python
@dataclass(frozen=True)
class CalculationProvenance:
    calculation_id: str
    timestamp: datetime
    calculator_name: str
    method_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    steps: Tuple[CalculationStep, ...]
    formula_references: Tuple[str, ...]
    provenance_hash: str  # SHA-256 of inputs + outputs + steps

    def verify_integrity(self) -> bool:
        """Verify provenance hash matches content."""
        computed = self._compute_hash()
        return computed == self.provenance_hash
```

---

## High-Level Architecture

### Component Diagram

```
+==============================================================================+
|                        GL-015 COMPONENT ARCHITECTURE                          |
+==============================================================================+
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  |                           API GATEWAY                                    |  |
|  |  +-------------+  +-------------+  +-------------+  +-------------+     |  |
|  |  | REST API    |  | WebSocket   |  | GraphQL     |  | gRPC        |     |  |
|  |  | (FastAPI)   |  | (Events)    |  | (Optional)  |  | (Internal)  |     |  |
|  |  +-------------+  +-------------+  +-------------+  +-------------+     |  |
|  +-------------------------------------------------------------------------+  |
|                                      |                                        |
|                                      v                                        |
|  +-------------------------------------------------------------------------+  |
|  |                         ORCHESTRATION LAYER                              |  |
|  |  +-------------------+  +-------------------+  +-------------------+     |  |
|  |  | Request Router    |  | Pipeline Manager  |  | Workflow Engine   |     |  |
|  |  |                   |  |                   |  |                   |     |  |
|  |  | - Authentication  |  | - Async Tasks     |  | - State Machine   |     |  |
|  |  | - Rate Limiting   |  | - Batch Handler   |  | - Event Sourcing  |     |  |
|  |  | - Validation      |  | - Priority Queue  |  | - Saga Pattern    |     |  |
|  |  +-------------------+  +-------------------+  +-------------------+     |  |
|  +-------------------------------------------------------------------------+  |
|                                      |                                        |
|                                      v                                        |
|  +-------------------------------------------------------------------------+  |
|  |                         CALCULATOR LAYER                                 |  |
|  |                    (Deterministic - Zero Hallucination)                  |  |
|  +-------------------------------------------------------------------------+  |
|  |                                                                         |  |
|  |  +--------------------+  +--------------------+  +-------------------+  |  |
|  |  | THERMAL IMAGE      |  | HEAT LOSS          |  | ECONOMIC          |  |  |
|  |  | ANALYZER           |  | CALCULATOR         |  | CALCULATOR        |  |  |
|  |  |                    |  |                    |  |                   |  |  |
|  |  | - Matrix Process   |  | - Conduction       |  | - ROI Analysis    |  |  |
|  |  | - Emissivity Corr  |  | - Convection       |  | - NPV/IRR         |  |  |
|  |  | - Hotspot Detect   |  | - Radiation        |  | - Payback         |  |  |
|  |  | - ROI Analysis     |  | - Combined         |  | - Cost Estimate   |  |  |
|  |  | - Anomaly Class    |  | - Surface Temp     |  | - Carbon Cost     |  |  |
|  |  | - Quality Assess   |  | - Annual Energy    |  |                   |  |  |
|  |  +--------------------+  +--------------------+  +-------------------+  |  |
|  |                                                                         |  |
|  |  +--------------------+  +--------------------+  +-------------------+  |  |
|  |  | SURFACE TEMP       |  | DEGRADATION        |  | REPAIR            |  |  |
|  |  | ANALYZER           |  | ASSESSOR           |  | PRIORITIZATION    |  |  |
|  |  |                    |  |                    |  |                   |  |  |
|  |  | - Iteration Solver |  | - Condition Grade  |  | - FMEA Scoring    |  |  |
|  |  | - Energy Balance   |  | - CUI Risk         |  | - Criticality     |  |  |
|  |  | - Convergence      |  | - Remaining Life   |  | - Work Scope      |  |  |
|  |  +--------------------+  +--------------------+  +-------------------+  |  |
|  |                                                                         |  |
|  +-------------------------------------------------------------------------+  |
|                                      |                                        |
|                                      v                                        |
|  +-------------------------------------------------------------------------+  |
|  |                         INTEGRATION LAYER                                |  |
|  +-------------------------------------------------------------------------+  |
|  |                                                                         |  |
|  |  +-----------------+  +-----------------+  +------------------------+  |  |
|  |  | CAMERA          |  | CMMS            |  | GREENLANG              |  |  |
|  |  | CONNECTORS      |  | CONNECTORS      |  | CONNECTORS             |  |  |
|  |  |                 |  |                 |  |                        |  |  |
|  |  | - FLIRConnector |  | - SAPPMConnector|  | - ThermosyncConnector  |  |  |
|  |  | - FlukeConnector|  | - MaximoConnect |  | - HeatreclaimConnector |  |  |
|  |  | - TestoConnector|  | - OracleEAMConn |  | - ExchangerProConnector|  |  |
|  |  | - OptrisConnect |  | - HexagonConnect|  |                        |  |  |
|  |  +-----------------+  +-----------------+  +------------------------+  |  |
|  |                                                                         |  |
|  +-------------------------------------------------------------------------+  |
|                                      |                                        |
|                                      v                                        |
|  +-------------------------------------------------------------------------+  |
|  |                         PERSISTENCE LAYER                                |  |
|  +-------------------------------------------------------------------------+  |
|  |                                                                         |  |
|  |  +------------------+  +------------------+  +---------------------+    |  |
|  |  | PostgreSQL       |  | Redis            |  | S3 Compatible       |    |  |
|  |  |                  |  |                  |  |                     |    |  |
|  |  | - Inspections    |  | - Session Cache  |  | - Thermal Images    |    |  |
|  |  | - Equipment      |  | - Calc Results   |  | - PDF Reports       |    |  |
|  |  | - Audit Trail    |  | - Rate Limits    |  | - Export Files      |    |  |
|  |  | - Provenance     |  | - API Tokens     |  | - Attachments       |    |  |
|  |  +------------------+  +------------------+  +---------------------+    |  |
|  |                                                                         |  |
|  +-------------------------------------------------------------------------+  |
|                                                                               |
+==============================================================================+
```

### Layer Responsibilities

| Layer | Responsibilities | Technologies |
|-------|------------------|--------------|
| **API Gateway** | Request handling, authentication, rate limiting | FastAPI, JWT |
| **Orchestration** | Workflow management, async processing | AsyncIO, Celery |
| **Calculator** | Deterministic calculations, provenance | Python Decimal |
| **Integration** | External system connectivity | HTTPX, OPC-UA |
| **Persistence** | Data storage, caching, file storage | PostgreSQL, Redis, S3 |

---

## Calculator Layer Design

### Calculator Architecture

The Calculator Layer is the core of INSULSCAN, implementing all engineering calculations with zero-hallucination guarantees.

```
+==============================================================================+
|                     CALCULATOR LAYER ARCHITECTURE                             |
+==============================================================================+
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                        CALCULATOR FACTORY                             |   |
|   |   (Manages calculator lifecycle and dependency injection)             |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|           +--------------------------|---------------------------+            |
|           |                          |                           |            |
|           v                          v                           v            |
|   +---------------+          +---------------+           +---------------+    |
|   | THERMAL       |          | HEAT LOSS     |           | ECONOMIC      |    |
|   | ANALYZERS     |          | CALCULATORS   |           | CALCULATORS   |    |
|   +---------------+          +---------------+           +---------------+    |
|   |               |          |               |           |               |    |
|   | ThermalImage  |          | HeatLoss      |           | Economic      |    |
|   | Analyzer      |          | Calculator    |           | Calculator    |    |
|   |   |           |          |   |           |           |   |           |    |
|   |   +- Matrix   |          |   +- Conduct  |           |   +- ROI      |    |
|   |   |  Process  |          |   |  ion      |           |   |  Analysis |    |
|   |   |           |          |   |           |           |   |           |    |
|   |   +- Hotspot  |          |   +- Convect  |           |   +- NPV/IRR  |    |
|   |   |  Detect   |          |   |  ion      |           |   |           |    |
|   |   |           |          |   |           |           |   +- Payback  |    |
|   |   +- ROI      |          |   +- Radiat   |           |   |           |    |
|   |   |  Analysis |          |   |  ion      |           |   +- Cost     |    |
|   |   |           |          |   |           |           |   |  Estimate |    |
|   |   +- Quality  |          |   +- Combined |           |               |    |
|   |   |  Assess   |          |   |           |           |               |    |
|   |   |           |          |   +- Surface  |           |               |    |
|   |   +- Anomaly  |          |   |  Temp     |           |               |    |
|   |      Class    |          |               |           |               |    |
|   +---------------+          +---------------+           +---------------+    |
|           |                          |                           |            |
|           +--------------------------|---------------------------+            |
|                                      |                                        |
|                                      v                                        |
|   +-----------------------------------------------------------------------+   |
|   |                        PROVENANCE MANAGER                             |   |
|   |   (Tracks all calculation steps with SHA-256 hashing)                 |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|                                      v                                        |
|   +-----------------------------------------------------------------------+   |
|   |                        CONSTANTS REGISTRY                             |   |
|   |   (Physical constants, material properties, lookup tables)            |   |
|   +-----------------------------------------------------------------------+   |
|                                                                               |
+==============================================================================+
```

### Calculator Class Hierarchy

```python
# Base calculator interface
class BaseCalculator(ABC):
    """Abstract base class for all calculators."""

    def __init__(self, precision: int = 6):
        self.precision = precision
        self._calculation_steps: List[CalculationStep] = []
        self._step_counter = 0

    @abstractmethod
    def calculate(self, **inputs) -> BaseResult:
        """Perform the calculation."""
        pass

    def _add_step(
        self,
        operation: str,
        description: str,
        inputs: Dict[str, Any],
        output_name: str,
        output_value: Any,
        formula: str = "",
        reference: str = ""
    ) -> None:
        """Record calculation step for provenance."""
        self._step_counter += 1
        step = CalculationStep(
            step_number=self._step_counter,
            operation=operation,
            description=description,
            inputs=inputs,
            output_name=output_name,
            output_value=output_value,
            formula=formula,
            reference=reference
        )
        self._calculation_steps.append(step)

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _quantize(self, value: Decimal, places: int = None) -> Decimal:
        """Round decimal to specified precision."""
        places = places or self.precision
        return value.quantize(
            Decimal(f"0.{'0' * places}"),
            rounding=ROUND_HALF_UP
        )
```

### Heat Loss Calculator Implementation

```python
class HeatLossCalculator(BaseCalculator):
    """
    ASTM C680 compliant heat loss calculator.

    Implements:
    - Conduction through insulation (Fourier's Law)
    - Natural and forced convection (Churchill-Chu)
    - Radiation heat transfer (Stefan-Boltzmann)
    - Combined heat transfer with iteration
    """

    def calculate_total_heat_loss(
        self,
        process_temperature_c: Decimal,
        ambient_temperature_c: Decimal,
        surface_temperature_c: Decimal,
        outer_diameter_m: Decimal,
        length_m: Decimal,
        geometry: SurfaceGeometry,
        surface_material: SurfaceMaterial,
        wind_speed_m_s: Decimal = Decimal("0.0")
    ) -> TotalHeatLossResult:
        """
        Calculate total heat loss combining all modes.

        Returns immutable result with complete provenance.
        """
        self._calculation_steps = []
        self._step_counter = 0

        # Calculate surface area
        area_m2 = self._calculate_surface_area(
            outer_diameter_m, length_m, geometry
        )

        # Calculate convection
        convection_result = self._calculate_convection(
            surface_temperature_c,
            ambient_temperature_c,
            outer_diameter_m,
            area_m2,
            geometry,
            wind_speed_m_s
        )

        # Calculate radiation
        radiation_result = self._calculate_radiation(
            surface_temperature_c,
            ambient_temperature_c,
            area_m2,
            surface_material
        )

        # Combine results
        total_heat_loss = (
            convection_result.heat_loss_w +
            radiation_result.heat_loss_w
        )

        # Calculate fractions
        conv_fraction = convection_result.heat_loss_w / total_heat_loss
        rad_fraction = radiation_result.heat_loss_w / total_heat_loss

        # Per unit metrics
        heat_loss_per_m = total_heat_loss / length_m
        heat_loss_per_m2 = total_heat_loss / area_m2

        # Overall U-value
        delta_t = surface_temperature_c - ambient_temperature_c
        u_value = total_heat_loss / (area_m2 * delta_t)

        # Compute provenance hash
        provenance_data = {
            "inputs": {
                "process_temperature_c": str(process_temperature_c),
                "ambient_temperature_c": str(ambient_temperature_c),
                "surface_temperature_c": str(surface_temperature_c),
                "outer_diameter_m": str(outer_diameter_m),
                "length_m": str(length_m),
                "geometry": geometry.name,
                "surface_material": surface_material.name,
                "wind_speed_m_s": str(wind_speed_m_s)
            },
            "outputs": {
                "total_heat_loss_w": str(total_heat_loss),
                "convection_fraction": str(conv_fraction),
                "radiation_fraction": str(rad_fraction)
            },
            "steps": [s.to_dict() for s in self._calculation_steps]
        }
        provenance_hash = self._compute_provenance_hash(provenance_data)

        return TotalHeatLossResult(
            total_heat_loss_w=self._quantize(total_heat_loss),
            total_heat_loss_w_per_m=self._quantize(heat_loss_per_m),
            total_heat_loss_w_per_m2=self._quantize(heat_loss_per_m2),
            convection_fraction=self._quantize(conv_fraction, 4),
            radiation_fraction=self._quantize(rad_fraction, 4),
            surface_temperature_c=surface_temperature_c,
            ambient_temperature_c=ambient_temperature_c,
            process_temperature_c=process_temperature_c,
            overall_u_value_w_m2_k=self._quantize(u_value),
            total_thermal_resistance_m2_k_per_w=self._quantize(
                Decimal("1") / u_value
            ),
            conduction_result=ConductionResult(...),
            convection_result=convection_result,
            radiation_result=radiation_result,
            iterations_to_converge=1,
            provenance_hash=provenance_hash
        )
```

### Convection Correlations

The calculator implements multiple convection correlations:

```python
class ConvectionCorrelations:
    """
    Churchill-Chu and related correlations for convection.

    References:
    - Churchill, S.W. & Chu, H.H.S. (1975)
    - ASTM C680-14
    - VDI Heat Atlas (2010)
    """

    @staticmethod
    def natural_convection_horizontal_cylinder(
        surface_temp_c: Decimal,
        ambient_temp_c: Decimal,
        diameter_m: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """
        Churchill-Chu correlation for horizontal cylinders.

        Nu = {0.6 + 0.387 * Ra^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27)}^2

        Valid for: 10^(-5) < Ra < 10^12
        """
        # Film temperature
        t_film_c = (surface_temp_c + ambient_temp_c) / 2
        t_film_k = t_film_c + KELVIN_OFFSET

        # Air properties at film temperature
        props = AirPropertiesTable.get_properties(float(t_film_k))

        # Temperature difference
        delta_t = abs(surface_temp_c - ambient_temp_c)

        # Grashof number: Gr = g * beta * delta_T * L^3 / nu^2
        beta = Decimal("1") / t_film_k  # Ideal gas approximation
        nu = Decimal(str(props['kinematic_viscosity']))

        gr = (GRAVITY * beta * delta_t * diameter_m**3) / nu**2

        # Prandtl number
        pr = Decimal(str(props['prandtl']))

        # Rayleigh number
        ra = gr * pr

        # Churchill-Chu correlation
        pr_term = (Decimal("0.559") / pr) ** (Decimal("9") / Decimal("16"))
        denom = (Decimal("1") + pr_term) ** (Decimal("8") / Decimal("27"))

        nu_term = Decimal("0.387") * (ra ** (Decimal("1") / Decimal("6"))) / denom
        nu_d = (Decimal("0.6") + nu_term) ** 2

        # Heat transfer coefficient
        k = Decimal(str(props['thermal_conductivity']))
        h = nu_d * k / diameter_m

        return h, nu_d

    @staticmethod
    def forced_convection_horizontal_cylinder(
        surface_temp_c: Decimal,
        ambient_temp_c: Decimal,
        diameter_m: Decimal,
        velocity_m_s: Decimal
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Hilpert correlation for forced convection over cylinders.

        Nu = C * Re^m * Pr^(1/3)

        Valid for: 0.4 < Re < 400,000
        """
        # Film temperature properties
        t_film_c = (surface_temp_c + ambient_temp_c) / 2
        t_film_k = t_film_c + KELVIN_OFFSET

        props = AirPropertiesTable.get_properties(float(t_film_k))

        # Reynolds number
        nu = Decimal(str(props['kinematic_viscosity']))
        re = velocity_m_s * diameter_m / nu

        # Prandtl number
        pr = Decimal(str(props['prandtl']))

        # Hilpert constants (Re-dependent)
        if re < Decimal("4"):
            c, m = Decimal("0.989"), Decimal("0.330")
        elif re < Decimal("40"):
            c, m = Decimal("0.911"), Decimal("0.385")
        elif re < Decimal("4000"):
            c, m = Decimal("0.683"), Decimal("0.466")
        elif re < Decimal("40000"):
            c, m = Decimal("0.193"), Decimal("0.618")
        else:
            c, m = Decimal("0.027"), Decimal("0.805")

        # Nusselt number
        nu_d = c * (re ** m) * (pr ** (Decimal("1") / Decimal("3")))

        # Heat transfer coefficient
        k = Decimal(str(props['thermal_conductivity']))
        h = nu_d * k / diameter_m

        return h, nu_d, re
```

### Thermal Image Analyzer Architecture

```python
class ThermalImageAnalyzer:
    """
    Zero-hallucination thermal image analyzer.

    Processing pipeline:
    1. Raw data conversion (radiometric to temperature)
    2. Emissivity correction
    3. Atmospheric compensation
    4. Hotspot detection
    5. Anomaly classification
    6. Quality assessment
    """

    def __init__(
        self,
        pixel_size_m: Optional[Decimal] = None,
        default_emissivity: Decimal = Decimal("0.95"),
        default_ambient_c: Decimal = Decimal("20.0")
    ):
        self.pixel_size_m = pixel_size_m
        self.default_emissivity = default_emissivity
        self.default_ambient_c = default_ambient_c
        self._calculation_steps: List[Dict[str, Any]] = []

    def analyze_thermal_image(
        self,
        raw_data: List[List[Union[int, float, Decimal]]],
        emissivity: Decimal = Decimal("0.95"),
        ambient_temperature_c: Optional[Decimal] = None,
        reflected_temperature_c: Decimal = Decimal("20.0"),
        distance_m: Decimal = Decimal("1.0"),
        relative_humidity: Decimal = Decimal("50.0"),
        rois: Optional[List[ROIDefinition]] = None,
        generate_map: bool = True,
        assess_quality: bool = True
    ) -> ThermalAnalysisResult:
        """
        Complete thermal image analysis pipeline.

        Returns:
            ThermalAnalysisResult with full provenance
        """
        start_time = time.perf_counter()
        analysis_id = str(uuid.uuid4())
        self._calculation_steps = []

        # Step 1: Process temperature matrix
        temperature_matrix, config = self.process_temperature_matrix(
            raw_data,
            emissivity=emissivity,
            reflected_temperature_c=reflected_temperature_c,
            distance_m=distance_m,
            relative_humidity=relative_humidity
        )

        # Step 2: Calculate ambient reference if not provided
        if ambient_temperature_c is None:
            ambient_temperature_c = self._calculate_ambient_reference(
                temperature_matrix
            )

        # Step 3: Calculate statistics
        statistics = self._calculate_statistics(temperature_matrix)

        # Step 4: Detect hotspots
        hotspots = self.detect_hotspots(
            temperature_matrix,
            ambient_temperature_c=ambient_temperature_c
        )

        # Step 5: Classify anomalies
        anomaly_classifications = []
        for hotspot in hotspots:
            classification = self.classify_thermal_anomaly(
                hotspot,
                ambient_temperature_c=ambient_temperature_c
            )
            anomaly_classifications.append(classification)

        # Step 6: Assess image quality
        image_quality = self.assess_image_quality(
            temperature_matrix,
            ambient_temperature_c=ambient_temperature_c
        ) if assess_quality else self._default_quality()

        # Step 7: Generate temperature map
        temperature_map = self.generate_temperature_map(
            temperature_matrix
        ) if generate_map else None

        # Step 8: Analyze ROIs
        roi_results = []
        if rois:
            for roi in rois:
                roi_result = self.analyze_roi(temperature_matrix, roi)
                roi_results.append(roi_result)

        # Compute processing time
        processing_time_ms = Decimal(str(
            (time.perf_counter() - start_time) * 1000
        ))

        # Compute provenance hash
        provenance_hash = self._compute_analysis_provenance(
            analysis_id, config, statistics, hotspots
        )

        return ThermalAnalysisResult(
            analysis_id=analysis_id,
            config=config,
            temperature_matrix_shape=(
                len(temperature_matrix),
                len(temperature_matrix[0]) if temperature_matrix else 0
            ),
            statistics=statistics,
            hotspots=tuple(hotspots),
            anomaly_classifications=tuple(anomaly_classifications),
            image_quality=image_quality,
            temperature_map=temperature_map,
            roi_results=tuple(roi_results),
            provenance_hash=provenance_hash,
            calculation_steps=tuple(self._calculation_steps),
            analysis_timestamp=datetime.now(timezone.utc),
            processing_time_ms=processing_time_ms.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        )
```

---

## Integration Layer Design

### Connector Architecture

```
+==============================================================================+
|                     INTEGRATION LAYER ARCHITECTURE                            |
+==============================================================================+
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                     CONNECTOR FACTORY                                 |   |
|   |   (Creates and manages connector instances with connection pooling)   |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|           +--------------------------|---------------------------+            |
|           |                          |                           |            |
|           v                          v                           v            |
|   +---------------+          +---------------+           +---------------+    |
|   | CAMERA        |          | CMMS          |           | WEATHER       |    |
|   | CONNECTORS    |          | CONNECTORS    |           | CONNECTORS    |    |
|   +---------------+          +---------------+           +---------------+    |
|   |               |          |               |           |               |    |
|   | +----------+  |          | +----------+  |           | +----------+  |    |
|   | |BaseCamera|  |          | |BaseCMMS  |  |           | |BaseWeather| |    |
|   | |Connector |  |          | |Connector |  |           | |Connector |  |    |
|   | +----+-----+  |          | +----+-----+  |           | +----+-----+  |    |
|   |      |        |          |      |        |           |      |        |    |
|   |      v        |          |      v        |           |      v        |    |
|   | +--------+    |          | +--------+    |           | +---------+   |    |
|   | |FLIR    |    |          | |SAP PM  |    |           | |OpenWeath|   |    |
|   | +--------+    |          | +--------+    |           | +---------+   |    |
|   | |Fluke   |    |          | |Maximo  |    |           | |NOAA     |   |    |
|   | +--------+    |          | +--------+    |           | +---------+   |    |
|   | |Testo   |    |          | |Oracle  |    |           |               |    |
|   | +--------+    |          | +--------+    |           |               |    |
|   | |Optris  |    |          | |Hexagon |    |           |               |    |
|   | +--------+    |          | +--------+    |           |               |    |
|   +---------------+          +---------------+           +---------------+    |
|                                                                               |
+==============================================================================+
```

### Base Connector Interface

```python
class BaseConnector(ABC):
    """
    Abstract base class for all external connectors.

    Provides:
    - Connection pooling
    - Retry with exponential backoff
    - Circuit breaker pattern
    - Metrics collection
    """

    def __init__(
        self,
        config: ConnectorConfig,
        pool_size: int = 10,
        timeout_seconds: int = 30
    ):
        self.config = config
        self.pool_size = pool_size
        self.timeout = timeout_seconds
        self._connection_pool: Optional[ConnectionPool] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=30,
            recovery_timeout=60
        )
        self._metrics = ConnectorMetrics()

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to external system."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check connection health."""
        pass

    async def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> Any:
        """Execute operation with exponential backoff retry."""
        last_exception = None

        for attempt in range(max_retries):
            try:
                if self._circuit_breaker.is_open:
                    raise CircuitBreakerOpen()

                result = await asyncio.wait_for(
                    operation(),
                    timeout=self.timeout
                )

                self._circuit_breaker.record_success()
                self._metrics.record_success()
                return result

            except Exception as e:
                last_exception = e
                self._circuit_breaker.record_failure()
                self._metrics.record_failure()

                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)

        raise last_exception
```

### CMMS Connector Implementation

```python
class SAPPMConnector(BaseCMMS Connector):
    """
    SAP Plant Maintenance connector.

    Integrates with SAP PM for:
    - Work order creation
    - Notification generation
    - Equipment master synchronization
    - Cost center posting
    """

    def __init__(
        self,
        host: str,
        client: str,
        username: str,
        password: str,
        **kwargs
    ):
        super().__init__(ConnectorConfig(host=host, **kwargs))
        self.client = client
        self.username = username
        self.password = password
        self._session: Optional[SAPSession] = None

    async def create_work_order(
        self,
        functional_location: str,
        order_type: str,
        priority: str,
        description: str,
        long_text: str,
        planned_start: date,
        estimated_cost: Decimal
    ) -> WorkOrderResult:
        """
        Create PM work order via BAPI.

        Uses: BAPI_ALM_ORDER_CREATE
        """
        bapi_input = {
            "ORDERDATA": {
                "ORDERID": "",  # Auto-generated
                "ORDER_TYPE": order_type,
                "FUNC_LOC": functional_location,
                "PRIORITY": priority,
                "SHORT_TEXT": description[:40],
                "BASIC_START": planned_start.strftime("%Y%m%d"),
                "PMACTTYPE": "002",  # Corrective maintenance
            },
            "OPERATION": [{
                "ACTIVITY": "0010",
                "DESCRIPTION": description,
                "WORK_CNTR": "INSULATION",
                "CALC_KEY": "1",
                "DURATION": "8",
                "DURATION_UNIT": "H"
            }],
            "COSTS": {
                "ESTIMATED_COST": str(estimated_cost),
                "CURRENCY": "USD"
            }
        }

        result = await self.execute_with_retry(
            lambda: self._call_bapi("BAPI_ALM_ORDER_CREATE", bapi_input)
        )

        return WorkOrderResult(
            order_number=result["RETURN"]["ORDER_NUMBER"],
            status="created",
            created_at=datetime.now(timezone.utc)
        )

    async def attach_inspection_report(
        self,
        order_number: str,
        report_path: str,
        document_type: str = "INS"
    ) -> None:
        """Attach inspection report to work order."""
        with open(report_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()

        bapi_input = {
            "OBJECTTYPE": "BUS2007",  # PM Order
            "OBJECTKEY": order_number,
            "DOCTYPE": document_type,
            "DOCFILE": content,
            "FILENAME": os.path.basename(report_path)
        }

        await self.execute_with_retry(
            lambda: self._call_bapi("BAPI_DOCUMENT_CREATE", bapi_input)
        )
```

---

## Data Flow Architecture

### Inspection Data Flow

```
+==============================================================================+
|                      INSPECTION DATA FLOW                                     |
+==============================================================================+
|                                                                               |
|   +---------------+                                                           |
|   | IR Camera     |                                                           |
|   | (FLIR/Fluke)  |                                                           |
|   +-------+-------+                                                           |
|           |                                                                   |
|           | Radiometric JPEG                                                  |
|           v                                                                   |
|   +-------+-------+    +---------------+                                      |
|   | Camera        |--->| S3 Blob       |                                      |
|   | Connector     |    | Storage       |                                      |
|   +-------+-------+    +---------------+                                      |
|           |                                                                   |
|           | Temperature Matrix                                                |
|           v                                                                   |
|   +-------+-------+                                                           |
|   | Input         |                                                           |
|   | Validator     |                                                           |
|   | (Pydantic)    |                                                           |
|   +-------+-------+                                                           |
|           |                                                                   |
|           | Validated InsulationInspectionInput                               |
|           v                                                                   |
|   +-------+-------+                                                           |
|   | Orchestrator  |                                                           |
|   | (Pipeline)    |                                                           |
|   +-------+-------+                                                           |
|           |                                                                   |
|           +--------+--------+--------+                                        |
|           |        |        |        |                                        |
|           v        v        v        v                                        |
|   +-------+--+ +---+----+ +-+------+ +--------+                               |
|   | Thermal  | | Heat   | | Degrad | | Repair |                               |
|   | Analyzer | | Loss   | | Assess | | Prior  |                               |
|   +-------+--+ +---+----+ +-+------+ +---+----+                               |
|           |        |        |            |                                    |
|           +--------+--------+------------+                                    |
|                    |                                                          |
|                    v                                                          |
|   +----------------+----------------+                                         |
|   | InsulationInspectionOutput      |                                         |
|   |                                 |                                         |
|   | - heat_loss_analysis            |                                         |
|   | - degradation_assessment        |                                         |
|   | - repair_priorities             |                                         |
|   | - economic_impact               |                                         |
|   | - provenance_hash               |                                         |
|   +----------------+----------------+                                         |
|                    |                                                          |
|           +--------+--------+--------+                                        |
|           |        |        |        |                                        |
|           v        v        v        v                                        |
|   +-------+--+ +---+----+ +-+------+ +--------+                               |
|   | Database | | Redis  | | Report | | CMMS   |                               |
|   | (Record) | | (Cache)| | (PDF)  | | (WO)   |                               |
|   +----------+ +--------+ +--------+ +--------+                               |
|                                                                               |
+==============================================================================+
```

### Event Flow (Event Sourcing Pattern)

```
+==============================================================================+
|                        EVENT SOURCING FLOW                                    |
+==============================================================================+
|                                                                               |
|   +-----------+     +------------+     +-------------+     +------------+     |
|   | Command   |---->| Command    |---->| Aggregate   |---->| Event      |     |
|   | Handler   |     | Validator  |     | Root        |     | Store      |     |
|   +-----------+     +------------+     +------+------+     +-----+------+     |
|                                              |                    |           |
|                                              |                    |           |
|   Commands:                                  |                    v           |
|   - StartInspection                          |              +-----+------+    |
|   - SubmitThermalImage                       |              | Event Bus  |    |
|   - ApproveRepair                            |              +-----+------+    |
|   - GenerateReport                           |                    |           |
|                                              |                    |           |
|                                              v                    v           |
|                                        +-----+------+       +----+-----+      |
|                                        | Domain     |       | Event    |      |
|                                        | Events     |       | Handlers |      |
|                                        +------------+       +----+-----+      |
|                                                                  |            |
|   Events:                                                        |            |
|   - InspectionStarted                                            |            |
|   - ThermalImageAnalyzed                                         v            |
|   - HeatLossCalculated                                     +-----+------+     |
|   - DegradationAssessed                                    | Projections|     |
|   - RepairRecommended                                      +------------+     |
|   - ReportGenerated                                                           |
|                                                                               |
+==============================================================================+
```

---

## Security Architecture

### Security Layers

```
+==============================================================================+
|                       SECURITY ARCHITECTURE                                   |
+==============================================================================+
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                        PERIMETER SECURITY                             |   |
|   |   - WAF (Web Application Firewall)                                    |   |
|   |   - DDoS Protection                                                   |   |
|   |   - Rate Limiting (100 req/min)                                       |   |
|   |   - IP Allowlisting (optional)                                        |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|                                      v                                        |
|   +-----------------------------------------------------------------------+   |
|   |                        API GATEWAY SECURITY                           |   |
|   |   - TLS 1.3 Termination                                               |   |
|   |   - JWT Token Validation                                              |   |
|   |   - OAuth2/OIDC Integration                                           |   |
|   |   - API Key Authentication                                            |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|                                      v                                        |
|   +-----------------------------------------------------------------------+   |
|   |                        APPLICATION SECURITY                           |   |
|   |   - RBAC (Role-Based Access Control)                                  |   |
|   |   - Input Validation (Pydantic)                                       |   |
|   |   - SQL Injection Prevention (SQLAlchemy)                             |   |
|   |   - XSS Prevention                                                    |   |
|   |   - CSRF Protection                                                   |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|                                      v                                        |
|   +-----------------------------------------------------------------------+   |
|   |                        DATA SECURITY                                  |   |
|   |   - Encryption at Rest (AES-256)                                      |   |
|   |   - Encryption in Transit (TLS 1.3)                                   |   |
|   |   - Database TDE                                                      |   |
|   |   - S3 Server-Side Encryption                                         |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|                                      v                                        |
|   +-----------------------------------------------------------------------+   |
|   |                        SECRETS MANAGEMENT                             |   |
|   |   - Kubernetes Secrets                                                |   |
|   |   - HashiCorp Vault (optional)                                        |   |
|   |   - Environment Variable Injection                                    |   |
|   |   - Secret Rotation                                                   |   |
|   +-----------------------------------------------------------------------+   |
|                                                                               |
+==============================================================================+
```

### Authentication Flow

```python
# JWT Authentication Flow
class JWTAuthenticator:
    """
    JWT Bearer token authentication.

    Token structure:
    {
        "sub": "user_id",
        "iss": "greenlang.io",
        "aud": "gl-015",
        "exp": 1234567890,
        "iat": 1234567800,
        "roles": ["inspector", "analyst"],
        "permissions": ["read:inspections", "create:inspections"]
    }
    """

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def authenticate(
        self,
        request: Request
    ) -> AuthenticatedUser:
        """Authenticate request and return user context."""
        # Extract token from header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise AuthenticationError("Missing or invalid token")

        token = auth_header[7:]

        try:
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience="gl-015",
                issuer="greenlang.io"
            )

            # Check expiration
            if payload["exp"] < time.time():
                raise AuthenticationError("Token expired")

            return AuthenticatedUser(
                user_id=payload["sub"],
                roles=payload["roles"],
                permissions=payload["permissions"]
            )

        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
```

### Authorization (RBAC)

```python
# Role-Based Access Control
class RBACAuthorizer:
    """
    RBAC authorization with permission checking.

    Roles:
    - viewer: Read-only access to inspections
    - inspector: Create and manage inspections
    - analyst: Run calculations, generate reports
    - admin: Full access including configuration
    """

    ROLE_PERMISSIONS = {
        "viewer": [
            "read:inspections",
            "read:reports"
        ],
        "inspector": [
            "read:inspections",
            "create:inspections",
            "upload:images",
            "read:reports"
        ],
        "analyst": [
            "read:inspections",
            "create:inspections",
            "upload:images",
            "run:calculations",
            "create:reports",
            "read:reports"
        ],
        "admin": [
            "*"  # All permissions
        ]
    }

    def authorize(
        self,
        user: AuthenticatedUser,
        required_permission: str
    ) -> bool:
        """Check if user has required permission."""
        for role in user.roles:
            permissions = self.ROLE_PERMISSIONS.get(role, [])
            if "*" in permissions or required_permission in permissions:
                return True
        return False

    def require_permission(self, permission: str):
        """Decorator to require permission for endpoint."""
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                user = request.state.user
                if not self.authorize(user, permission):
                    raise AuthorizationError(
                        f"Permission '{permission}' required"
                    )
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator
```

---

## Scalability Considerations

### Horizontal Scaling Architecture

```
+==============================================================================+
|                    HORIZONTAL SCALING ARCHITECTURE                            |
+==============================================================================+
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                        LOAD BALANCER (L7)                             |   |
|   |   - Round-robin / Least connections                                   |   |
|   |   - Health check routing                                              |   |
|   |   - Session affinity (optional)                                       |   |
|   +-----------------------------------+-----------------------------------+   |
|                                       |                                       |
|           +---------------------------+---------------------------+           |
|           |                           |                           |           |
|           v                           v                           v           |
|   +---------------+           +---------------+           +---------------+   |
|   | GL-015        |           | GL-015        |           | GL-015        |   |
|   | Instance 1    |           | Instance 2    |           | Instance N    |   |
|   |               |           |               |           |               |   |
|   | - API Server  |           | - API Server  |           | - API Server  |   |
|   | - Calculators |           | - Calculators |           | - Calculators |   |
|   +-------+-------+           +-------+-------+           +-------+-------+   |
|           |                           |                           |           |
|           +---------------------------+---------------------------+           |
|                                       |                                       |
|                                       v                                       |
|   +-----------------------------------------------------------------------+   |
|   |                        SHARED RESOURCES                               |   |
|   +-----------------------------------------------------------------------+   |
|   |                                                                       |   |
|   |  +------------------+  +------------------+  +------------------+     |   |
|   |  | PostgreSQL       |  | Redis Cluster    |  | S3 Storage       |     |   |
|   |  | (Primary/Replica)|  | (3+ nodes)       |  | (Multi-region)   |     |   |
|   |  +------------------+  +------------------+  +------------------+     |   |
|   |                                                                       |   |
|   +-----------------------------------------------------------------------+   |
|                                                                               |
+==============================================================================+
```

### Auto-Scaling Configuration

```yaml
# Kubernetes HPA Configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-015-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-015-insulscan
  minReplicas: 2
  maxReplicas: 10
  metrics:
    # CPU-based scaling
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # Memory-based scaling
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    # Custom metric: request queue depth
    - type: Pods
      pods:
        metric:
          name: request_queue_depth
        target:
          type: AverageValue
          averageValue: 10
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
```

### Batch Processing for Scale

```python
class BatchProcessor:
    """
    Batch processor for large-scale inspections.

    Features:
    - Parallel processing with worker pool
    - Progress tracking
    - Partial failure handling
    - Result aggregation
    """

    def __init__(
        self,
        max_workers: int = 4,
        chunk_size: int = 25,
        max_batch_size: int = 100
    ):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size
        self._executor = ProcessPoolExecutor(max_workers=max_workers)

    async def process_batch(
        self,
        images: List[ThermalImageData],
        ambient_conditions: AmbientConditions,
        equipment_list: List[EquipmentParameters],
        insulation_specs: List[InsulationSpecifications]
    ) -> BatchResult:
        """Process batch of thermal images in parallel."""
        if len(images) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(images)} exceeds maximum {self.max_batch_size}"
            )

        # Create chunks for parallel processing
        chunks = list(self._create_chunks(
            zip(images, equipment_list, insulation_specs),
            self.chunk_size
        ))

        results = []
        errors = []

        # Process chunks in parallel
        futures = [
            self._process_chunk(chunk, ambient_conditions)
            for chunk in chunks
        ]

        for future in asyncio.as_completed(futures):
            try:
                chunk_results = await future
                results.extend(chunk_results)
            except Exception as e:
                errors.append(str(e))

        return BatchResult(
            total_processed=len(results),
            successful=len([r for r in results if r.success]),
            failed=len(errors),
            results=results,
            errors=errors
        )

    def _create_chunks(self, iterable, size):
        """Split iterable into chunks."""
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) >= size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
```

---

## Deployment Architecture

### Multi-Environment Deployment

```
+==============================================================================+
|                     MULTI-ENVIRONMENT ARCHITECTURE                            |
+==============================================================================+
|                                                                               |
|   +---------------------------+                                               |
|   |    DEVELOPMENT            |                                               |
|   |    (Local/Docker)         |                                               |
|   |                           |                                               |
|   |  - Single instance        |                                               |
|   |  - SQLite/PostgreSQL      |                                               |
|   |  - Mock integrations      |                                               |
|   |  - Debug logging          |                                               |
|   +---------------------------+                                               |
|               |                                                               |
|               | CI/CD Pipeline                                                |
|               v                                                               |
|   +---------------------------+                                               |
|   |    STAGING                |                                               |
|   |    (Kubernetes)           |                                               |
|   |                           |                                               |
|   |  - 2 replicas             |                                               |
|   |  - PostgreSQL (shared)    |                                               |
|   |  - Redis (single)         |                                               |
|   |  - Test integrations      |                                               |
|   |  - Info logging           |                                               |
|   +---------------------------+                                               |
|               |                                                               |
|               | Promotion Gate                                                |
|               v                                                               |
|   +---------------------------+                                               |
|   |    PRODUCTION             |                                               |
|   |    (Kubernetes HA)        |                                               |
|   |                           |                                               |
|   |  - 2-8 replicas (HPA)     |                                               |
|   |  - PostgreSQL HA cluster  |                                               |
|   |  - Redis cluster (3+)     |                                               |
|   |  - Production integrations|                                               |
|   |  - Warning logging        |                                               |
|   |  - Full monitoring        |                                               |
|   +---------------------------+                                               |
|                                                                               |
+==============================================================================+
```

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yaml
name: Deploy GL-015

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run linting
        run: |
          ruff check .
          black --check .
          mypy .
      - name: Run tests
        run: pytest tests/ -v --cov=gl_015 --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: |
          docker build -t greenlang/gl-015:${{ github.sha }} .
      - name: Push to registry
        run: |
          docker push greenlang/gl-015:${{ github.sha }}

  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/gl-015 \
            gl-015=greenlang/gl-015:${{ github.sha }} \
            --namespace=staging

  deploy-production:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/gl-015 \
            gl-015=greenlang/gl-015:${{ github.sha }} \
            --namespace=production
```

---

## Database Schema

### Entity Relationship Diagram

```
+==============================================================================+
|                        DATABASE SCHEMA (ERD)                                  |
+==============================================================================+
|                                                                               |
|   +------------------+         +------------------+                           |
|   | equipment        |         | inspections      |                           |
|   +------------------+         +------------------+                           |
|   | PK id            |<-----+  | PK id            |                           |
|   | equipment_tag    |      |  | FK equipment_id  |----+                      |
|   | equipment_type   |      |  | inspection_date  |    |                      |
|   | location         |      |  | inspector_id     |    |                      |
|   | process_temp_c   |      |  | status           |    |                      |
|   | created_at       |      |  | created_at       |    |                      |
|   +------------------+      |  +------------------+    |                      |
|                             |          |               |                      |
|   +------------------+      |          |               |                      |
|   | insulation_specs |      |          |               |                      |
|   +------------------+      |          |               |                      |
|   | PK id            |      |          |               |                      |
|   | FK equipment_id  |------+          |               |                      |
|   | insulation_type  |                 |               |                      |
|   | thickness_mm     |                 |               |                      |
|   | jacket_type      |                 |               |                      |
|   | install_date     |                 |               |                      |
|   +------------------+                 |               |                      |
|                                        |               |                      |
|   +------------------+                 |               |                      |
|   | thermal_images   |<----------------+               |                      |
|   +------------------+                                 |                      |
|   | PK id            |                                 |                      |
|   | FK inspection_id |                                 |                      |
|   | image_url        |                                 |                      |
|   | camera_type      |                                 |                      |
|   | capture_time     |                                 |                      |
|   | temp_matrix_ref  |                                 |                      |
|   +------------------+                                 |                      |
|                                                        |                      |
|   +------------------+                                 |                      |
|   | analysis_results |<--------------------------------+                      |
|   +------------------+                                                        |
|   | PK id            |                                                        |
|   | FK inspection_id |                                                        |
|   | heat_loss_w      |                                                        |
|   | condition_score  |                                                        |
|   | cui_risk_level   |                                                        |
|   | recommended_action|                                                       |
|   | provenance_hash  |                                                        |
|   +------------------+                                                        |
|                                                                               |
|   +------------------+         +------------------+                           |
|   | audit_trail      |         | provenance_records|                          |
|   +------------------+         +------------------+                           |
|   | PK id            |         | PK id            |                           |
|   | entity_type      |         | FK analysis_id   |                           |
|   | entity_id        |         | calculator_name  |                           |
|   | action           |         | method_name      |                           |
|   | user_id          |         | inputs_json      |                           |
|   | timestamp        |         | outputs_json     |                           |
|   | changes_json     |         | steps_json       |                           |
|   +------------------+         | hash             |                           |
|                                +------------------+                           |
|                                                                               |
+==============================================================================+
```

### SQL Schema Definition

```sql
-- Equipment table
CREATE TABLE equipment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_tag VARCHAR(50) NOT NULL UNIQUE,
    equipment_type VARCHAR(20) NOT NULL,
    location VARCHAR(200),
    unit_code VARCHAR(50),
    process_temp_c DECIMAL(8,2),
    design_temp_c DECIMAL(8,2),
    criticality VARCHAR(10) DEFAULT 'medium',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT chk_equipment_type CHECK (
        equipment_type IN ('pipe', 'vessel', 'tank', 'exchanger',
                          'column', 'duct', 'valve', 'flange')
    )
);

-- Inspections table
CREATE TABLE inspections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_id UUID NOT NULL REFERENCES equipment(id),
    inspection_date DATE NOT NULL,
    inspector_id VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    ambient_temp_c DECIMAL(6,2),
    wind_speed_m_s DECIMAL(5,2),
    sky_condition VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT chk_status CHECK (
        status IN ('pending', 'processing', 'completed', 'failed')
    )
);

-- Analysis results table
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inspection_id UUID NOT NULL REFERENCES inspections(id),
    heat_loss_w DECIMAL(12,2),
    heat_loss_w_per_m DECIMAL(10,2),
    annual_energy_mwh DECIMAL(10,3),
    annual_cost_usd DECIMAL(12,2),
    condition_score DECIMAL(5,2),
    overall_condition VARCHAR(20),
    degradation_severity VARCHAR(20),
    cui_risk_level VARCHAR(20),
    cui_risk_score DECIMAL(5,2),
    remaining_life_years DECIMAL(5,2),
    recommended_action VARCHAR(30),
    priority_ranking INTEGER,
    repair_cost_usd DECIMAL(12,2),
    payback_months DECIMAL(6,2),
    roi_percent DECIMAL(8,2),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_inspection_id (inspection_id),
    INDEX idx_condition (overall_condition),
    INDEX idx_priority (priority_ranking)
);

-- Provenance records for audit
CREATE TABLE provenance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES analysis_results(id),
    calculator_name VARCHAR(100) NOT NULL,
    method_name VARCHAR(100) NOT NULL,
    inputs_json JSONB NOT NULL,
    outputs_json JSONB NOT NULL,
    steps_json JSONB NOT NULL,
    formula_references TEXT[],
    hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_analysis (analysis_id),
    INDEX idx_hash (hash)
);

-- Audit trail
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(20) NOT NULL,
    user_id VARCHAR(50),
    ip_address INET,
    user_agent TEXT,
    changes_json JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_entity (entity_type, entity_id),
    INDEX idx_timestamp (timestamp)
);
```

---

## Observability Architecture

### Metrics, Logging, and Tracing

```
+==============================================================================+
|                     OBSERVABILITY ARCHITECTURE                                |
+==============================================================================+
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                          GL-015 APPLICATION                           |   |
|   +-----------------------------------------------------------------------+   |
|           |                           |                           |           |
|           | Metrics                   | Logs                      | Traces    |
|           v                           v                           v           |
|   +---------------+           +---------------+           +---------------+   |
|   | Prometheus    |           | Loki          |           | Jaeger        |   |
|   | Metrics       |           | Log           |           | Tracing       |   |
|   | Collector     |           | Aggregator    |           | Backend       |   |
|   +-------+-------+           +-------+-------+           +-------+-------+   |
|           |                           |                           |           |
|           +---------------------------+---------------------------+           |
|                                       |                                       |
|                                       v                                       |
|                              +--------+--------+                              |
|                              |    Grafana      |                              |
|                              |    Dashboard    |                              |
|                              +-----------------+                              |
|                                       |                                       |
|                                       v                                       |
|                              +--------+--------+                              |
|                              | Alert Manager   |                              |
|                              | (PagerDuty)     |                              |
|                              +-----------------+                              |
|                                                                               |
+==============================================================================+
```

### Key Metrics

```python
# Prometheus metrics definition
from prometheus_client import Counter, Histogram, Gauge

# Inspection metrics
inspections_total = Counter(
    'gl015_inspections_total',
    'Total number of inspections',
    ['location', 'condition', 'status']
)

inspection_duration = Histogram(
    'gl015_inspection_duration_seconds',
    'Inspection processing duration',
    ['calculator'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
)

# Heat loss metrics
heat_loss_gauge = Gauge(
    'gl015_heat_loss_w_m2',
    'Current heat loss rate',
    ['equipment_id', 'location']
)

condition_score_gauge = Gauge(
    'gl015_condition_score',
    'Insulation condition score',
    ['equipment_id', 'location']
)

# CUI risk metrics
cui_risk_gauge = Gauge(
    'gl015_cui_risk_score',
    'Corrosion Under Insulation risk score',
    ['equipment_id', 'location']
)

# API metrics
api_requests = Counter(
    'gl015_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_latency = Histogram(
    'gl015_api_latency_seconds',
    'API request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
)
```

---

## Disaster Recovery

### Backup and Recovery Strategy

```
+==============================================================================+
|                     DISASTER RECOVERY ARCHITECTURE                            |
+==============================================================================+
|                                                                               |
|   PRIMARY REGION (us-east-1)                                                  |
|   +-----------------------------------------------------------------------+   |
|   |  +---------------+  +---------------+  +---------------+              |   |
|   |  | GL-015        |  | PostgreSQL    |  | S3            |              |   |
|   |  | Cluster       |  | Primary       |  | Primary       |              |   |
|   |  +-------+-------+  +-------+-------+  +-------+-------+              |   |
|   |          |                  |                  |                      |   |
|   +----------|------------------|------------------|----------------------+   |
|              |                  |                  |                          |
|              | State            | WAL              | Cross-Region              |
|              | Sync             | Shipping         | Replication               |
|              |                  |                  |                          |
|              v                  v                  v                          |
|   DR REGION (us-west-2)                                                       |
|   +-----------------------------------------------------------------------+   |
|   |  +---------------+  +---------------+  +---------------+              |   |
|   |  | GL-015        |  | PostgreSQL    |  | S3            |              |   |
|   |  | Standby       |  | Replica       |  | Replica       |              |   |
|   |  +---------------+  +---------------+  +---------------+              |   |
|   +-----------------------------------------------------------------------+   |
|                                                                               |
|   RECOVERY OBJECTIVES:                                                        |
|   - RPO (Recovery Point Objective): < 1 hour                                  |
|   - RTO (Recovery Time Objective): < 4 hours                                  |
|                                                                               |
+==============================================================================+
```

### Backup Schedule

| Component | Backup Type | Frequency | Retention |
|-----------|-------------|-----------|-----------|
| PostgreSQL | Full | Daily | 30 days |
| PostgreSQL | Incremental (WAL) | Continuous | 7 days |
| Redis | RDB Snapshot | Hourly | 24 hours |
| S3 Images | Cross-region replication | Real-time | Indefinite |
| Configuration | GitOps | On change | Forever |

---

## Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| **CUI** | Corrosion Under Insulation |
| **FMEA** | Failure Mode and Effects Analysis |
| **Heat Flux** | Heat transfer per unit area (W/m2) |
| **NPV** | Net Present Value |
| **ROI** | Return on Investment |
| **RUL** | Remaining Useful Life |
| **ASTM C680** | Standard Practice for Heat Loss from Insulated Pipe |
| **ASTM C1055** | Standard Guide for Heated System Surface Conditions |

### B. Reference Standards

| Standard | Title |
|----------|-------|
| ASTM C680 | Standard Practice for Estimate of Heat Gain or Loss |
| ASTM C1055 | Standard Guide for Heated System Surface Conditions |
| ASTM E1934 | Examining Equipment with Infrared Thermography |
| ISO 12241 | Thermal Insulation for Building Equipment |
| VDI 2055 | Thermal Insulation for Heated and Refrigerated Equipment |
| NACE SP0198 | Control of Corrosion Under Insulation |

### C. Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-01 | GL-TechWriter | Initial release |

---

**Document Classification: Internal - Technical Documentation**

**Copyright 2025 GreenLang AI Agent Factory. All rights reserved.**
