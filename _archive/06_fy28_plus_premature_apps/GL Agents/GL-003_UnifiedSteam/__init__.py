"""
GL-003 UNIFIEDSTEAM - Steam System Optimizer

Unified steam optimizer with IAPWS-IF97 thermodynamic calculations, SHAP/LIME
explainability, causal inference for root cause analysis, steam quality control,
desuperheater optimization, enthalpy balance, condensate recovery, and real-time
uncertainty quantification.

Agent Details:
    - Agent ID: GL-003
    - Codename: UNIFIEDSTEAM
    - Domain: Steam Systems
    - Type: Optimizer / Advisory + Control Support
    - Priority: High (P1)
    - Business Value: $14B
    - Target: Q1 2026
    - Status: Implemented

Absorbed Capabilities:
    - GL-008: Steam trap diagnostics + loss estimation
    - GL-012: Desuperheater optimization + steam conditioning
    - GL-017: Condensate recovery optimization + return KPIs

Key Features:
    - IAPWS-IF97 thermodynamic property calculations
    - Real-time steam properties (enthalpy, entropy, density, quality, superheat)
    - Enthalpy and mass balances across steam headers
    - Desuperheater operation optimization
    - Steam quality monitoring and control
    - Steam trap condition prediction (predictive failure using acoustics)
    - Causal root-cause analysis (RCA) for deviations
    - SHAP/LIME explainability for all recommendations
    - Uncertainty quantification with 95% confidence bounds
    - Climate/energy impact estimates (fuel, cost, CO2e)

Inputs:
    - Steam pressure, temperature, flow, quality
    - Condensate return flow + temperature
    - Water chemistry (conductivity, pH, dissolved oxygen, silica)
    - Steam trap acoustics (edge features)
    - Condenser vacuum, cooling water temperature
    - Operational context (boiler load, PRV status, valve positions)

Outputs:
    - Steam quality metrics + alarms
    - Desuperheater spray-water setpoint recommendations
    - Enthalpy-balance dashboards (losses, heat rate, KPIs)
    - Optimization recommendations (operational + maintenance)
    - Explainability reports (why + contributing signals)
    - Causal analysis (ranked root causes + counterfactuals)
    - Uncertainty bounds for key computed metrics
    - Trap failure predictions + maintenance priority list

Integrations:
    - OPC-UA (OT data acquisition)
    - Kafka (streaming transport)
    - gRPC (internal service calls)
    - GraphQL (client-facing API)
    - Steam meters, pressure sensors, temperature probes, quality analyzers

Author: GreenLang AI Agent Workforce
Version: 1.0.0
License: Proprietary - GreenLang
"""

__version__ = "1.0.0"
__author__ = "GreenLang AI Agent Workforce"
__agent_id__ = "GL-003"
__codename__ = "UNIFIEDSTEAM"

# Core components
from .core import (
    SteamSystemConfig,
    SteamProcessData,
    SteamProperties,
    SteamSystemOrchestrator,
)

# Thermodynamics engine
from .thermodynamics import (
    compute_properties,
    get_saturation_properties,
    detect_steam_state,
    compute_superheat_degree,
    compute_dryness_fraction,
    compute_mass_balance,
    compute_energy_balance,
)

# Calculators
from .calculators import (
    DesuperheaterCalculator,
    CondensateCalculator,
    TrapDiagnosticsCalculator,
    HeatBalanceCalculator,
    SteamKPICalculator,
)

# Optimization
from .optimization import (
    DesuperheaterOptimizer,
    CondensateRecoveryOptimizer,
    SteamNetworkOptimizer,
    TrapMaintenanceOptimizer,
    RecommendationEngine,
)

# Explainability
from .explainability import (
    PhysicsExplainer,
    SHAPExplainer,
    LIMEExplainer,
    ExplanationGenerator,
)

# Causal inference
from .causal import (
    CausalGraph,
    RootCauseAnalyzer,
    CounterfactualEngine,
    InterventionRecommender,
)

# Control
from .control import (
    DesuperheaterController,
    SteamQualityController,
    PRVController,
    SetpointManager,
)

# Safety
from .safety import (
    SafetyEnvelope,
    ConstraintValidator,
    InterlockManager,
)

# Monitoring
from .monitoring import (
    AlertManager,
    HealthMonitor,
    MetricsCollector,
)

# Audit
from .audit import (
    AuditLogger,
    ProvenanceTracker,
    EvidencePackager,
)

# Uncertainty
from .uncertainty import (
    SensorUncertaintyManager,
    UncertaintyPropagator,
    UncertaintyGate,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__codename__",
    # Core
    "SteamSystemConfig",
    "SteamProcessData",
    "SteamProperties",
    "SteamSystemOrchestrator",
    # Thermodynamics
    "compute_properties",
    "get_saturation_properties",
    "detect_steam_state",
    "compute_superheat_degree",
    "compute_dryness_fraction",
    "compute_mass_balance",
    "compute_energy_balance",
    # Calculators
    "DesuperheaterCalculator",
    "CondensateCalculator",
    "TrapDiagnosticsCalculator",
    "HeatBalanceCalculator",
    "SteamKPICalculator",
    # Optimization
    "DesuperheaterOptimizer",
    "CondensateRecoveryOptimizer",
    "SteamNetworkOptimizer",
    "TrapMaintenanceOptimizer",
    "RecommendationEngine",
    # Explainability
    "PhysicsExplainer",
    "SHAPExplainer",
    "LIMEExplainer",
    "ExplanationGenerator",
    # Causal
    "CausalGraph",
    "RootCauseAnalyzer",
    "CounterfactualEngine",
    "InterventionRecommender",
    # Control
    "DesuperheaterController",
    "SteamQualityController",
    "PRVController",
    "SetpointManager",
    # Safety
    "SafetyEnvelope",
    "ConstraintValidator",
    "InterlockManager",
    # Monitoring
    "AlertManager",
    "HealthMonitor",
    "MetricsCollector",
    # Audit
    "AuditLogger",
    "ProvenanceTracker",
    "EvidencePackager",
    # Uncertainty
    "SensorUncertaintyManager",
    "UncertaintyPropagator",
    "UncertaintyGate",
]
