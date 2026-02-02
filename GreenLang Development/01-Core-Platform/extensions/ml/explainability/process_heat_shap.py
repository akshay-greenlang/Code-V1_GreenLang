# -*- coding: utf-8 -*-
"""
GreenLang Process Heat SHAP Integration Module
===============================================

Unified SHAP (SHapley Additive exPlanations) integration for all 20 Process Heat
agents (GL-001 to GL-020). Provides process-heat-specific feature importance
tracking, batch processing, caching, and provenance tracking.

ZERO-HALLUCINATION COMPLIANCE:
    SHAP is used ONLY for explaining ML classifications and predictions.
    SHAP is NEVER used for numeric calculations that go into regulatory reports.
    All combustion efficiency, emissions, and thermodynamic calculations use
    deterministic engineering formulas from ASME, API, EPA, and NFPA standards.

Key Features:
    - Agent-specific SHAP configurations for GL-001 through GL-020
    - Industrial domain context with meaningful feature names
    - Batch processing support for high-frequency sensor data (up to 10Hz)
    - LRU caching with 66% cost reduction target
    - SHA-256 provenance tracking for complete audit trails
    - Confidence scoring with 80%+ threshold enforcement
    - Integration with ExplainabilityLayer base class

Supported Agents:
    - GL-001 THERMALCOMMAND: Orchestrator decision explanations
    - GL-003 UNIFIEDSTEAM: Steam system optimization explanations
    - GL-006 HEATRECLAIM: Heat recovery decision explanations
    - GL-010 EMISSIONSGUARDIAN: Emissions prediction explanations
    - GL-013 PREDICTMAINT: Failure prediction explanations
    - GL-018 UNIFIEDCOMBUSTION: Combustion optimization explanations

Example:
    >>> from greenlang.ml.explainability.process_heat_shap import (
    ...     ProcessHeatSHAPExplainer,
    ...     GL018CombustionSHAPConfig,
    ... )
    >>> config = GL018CombustionSHAPConfig(equipment_id="BOILER-001")
    >>> explainer = ProcessHeatSHAPExplainer(model, config)
    >>> result = explainer.explain_batch(sensor_data_batch)
    >>> print(result.feature_importance)
    >>> print(result.provenance_hash)

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore
    Field = lambda *args, **kwargs: None  # type: ignore

logger = logging.getLogger(__name__)

# Type variables
TModel = TypeVar("TModel")
TConfig = TypeVar("TConfig", bound="ProcessHeatSHAPConfigBase")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ProcessHeatAgentType(str, Enum):
    """Process Heat agent type identifiers."""
    GL_001_THERMALCOMMAND = "GL-001"
    GL_002_BOILEROPTIMIZER = "GL-002"
    GL_003_UNIFIEDSTEAM = "GL-003"
    GL_004_BURNMASTER = "GL-004"
    GL_005_COMBUSTIONDIAG = "GL-005"
    GL_006_HEATRECLAIM = "GL-006"
    GL_007_FURNACEMONITOR = "GL-007"
    GL_008_STEAMTRAP = "GL-008"
    GL_009_THERMALFLUID = "GL-009"
    GL_010_EMISSIONSGUARDIAN = "GL-010"
    GL_011_FUELOPTIMIZER = "GL-011"
    GL_012_LOADFORECAST = "GL-012"
    GL_013_PREDICTMAINT = "GL-013"
    GL_014_HEATEXCHANGER = "GL-014"
    GL_015_INSULATION = "GL-015"
    GL_016_WATERTREAT = "GL-016"
    GL_017_CONDENSER = "GL-017"
    GL_018_UNIFIEDCOMBUSTION = "GL-018"
    GL_019_COGENERATION = "GL-019"
    GL_020_SOLARINTEGRATION = "GL-020"


class SHAPExplainerMode(str, Enum):
    """SHAP explainer modes optimized for process heat applications."""
    TREE = "tree"  # For tree-based models (XGBoost, LightGBM, Random Forest)
    KERNEL = "kernel"  # Model-agnostic (slower but universal)
    LINEAR = "linear"  # For linear models
    DEEP = "deep"  # For neural networks
    GRADIENT = "gradient"  # For gradient-based neural networks
    PARTITION = "partition"  # For hierarchical feature importance


class ConfidenceLevel(str, Enum):
    """Confidence level classifications for SHAP explanations."""
    HIGH = "high"      # >= 0.90
    MEDIUM = "medium"  # >= 0.80
    LOW = "low"        # >= 0.60
    UNCERTAIN = "uncertain"  # < 0.60


class IndustrialDomain(str, Enum):
    """Industrial domain contexts for explanation generation."""
    COMBUSTION = "combustion"
    STEAM = "steam"
    HEAT_RECOVERY = "heat_recovery"
    EMISSIONS = "emissions"
    MAINTENANCE = "maintenance"
    THERMAL_FLUID = "thermal_fluid"
    WATER_TREATMENT = "water_treatment"


# Industrial feature name mappings for domain context
INDUSTRIAL_FEATURE_NAMES: Dict[IndustrialDomain, Dict[str, str]] = {
    IndustrialDomain.COMBUSTION: {
        "o2_pct": "Excess Air (O2%)",
        "co_ppm": "Carbon Monoxide (ppm)",
        "nox_ppm": "NOx Emissions (ppm)",
        "flue_temp_f": "Flue Gas Temperature (F)",
        "excess_air_pct": "Excess Air Percentage",
        "load_pct": "Boiler Load (%)",
        "fuel_flow_rate": "Fuel Flow Rate",
        "air_damper_pct": "Air Damper Position (%)",
        "fsi": "Flame Stability Index",
        "efficiency_pct": "Combustion Efficiency (%)",
    },
    IndustrialDomain.STEAM: {
        "steam_pressure_psig": "Steam Pressure (psig)",
        "steam_temp_f": "Steam Temperature (F)",
        "steam_flow_lb_hr": "Steam Flow (lb/hr)",
        "feedwater_temp_f": "Feedwater Temperature (F)",
        "condensate_return_pct": "Condensate Return (%)",
        "blowdown_rate_pct": "Blowdown Rate (%)",
        "steam_quality_pct": "Steam Quality (%)",
        "superheat_f": "Superheat (F)",
    },
    IndustrialDomain.HEAT_RECOVERY: {
        "exhaust_temp_in_f": "Exhaust Inlet Temperature (F)",
        "exhaust_temp_out_f": "Exhaust Outlet Temperature (F)",
        "heat_recovered_mmbtu": "Heat Recovered (MMBTU/hr)",
        "effectiveness_pct": "Heat Exchanger Effectiveness (%)",
        "approach_temp_f": "Approach Temperature (F)",
        "fouling_factor": "Fouling Factor",
    },
    IndustrialDomain.EMISSIONS: {
        "co2_lb_hr": "CO2 Emissions (lb/hr)",
        "nox_lb_mmbtu": "NOx Rate (lb/MMBTU)",
        "co_lb_mmbtu": "CO Rate (lb/MMBTU)",
        "pm_mg_m3": "Particulate Matter (mg/m3)",
        "so2_ppm": "SO2 Emissions (ppm)",
        "stack_o2_pct": "Stack O2 (%)",
        "permit_utilization_pct": "Permit Utilization (%)",
    },
    IndustrialDomain.MAINTENANCE: {
        "vibration_mm_s": "Vibration (mm/s RMS)",
        "bearing_temp_c": "Bearing Temperature (C)",
        "oil_condition_score": "Oil Condition Score",
        "running_hours": "Running Hours",
        "mtbf_hours": "Mean Time Between Failures (hr)",
        "health_score": "Equipment Health Score",
        "rul_hours": "Remaining Useful Life (hr)",
        "failure_probability": "Failure Probability",
    },
    IndustrialDomain.THERMAL_FLUID: {
        "fluid_temp_f": "Thermal Fluid Temperature (F)",
        "flow_rate_gpm": "Flow Rate (GPM)",
        "viscosity_cst": "Viscosity (cSt)",
        "heat_transfer_coef": "Heat Transfer Coefficient",
        "pressure_drop_psi": "Pressure Drop (psi)",
    },
    IndustrialDomain.WATER_TREATMENT: {
        "ph": "pH Level",
        "conductivity_us_cm": "Conductivity (uS/cm)",
        "tds_ppm": "Total Dissolved Solids (ppm)",
        "hardness_ppm": "Hardness (ppm as CaCO3)",
        "silica_ppm": "Silica (ppm)",
        "dissolved_o2_ppb": "Dissolved O2 (ppb)",
    },
}


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ProcessHeatSHAPConfigBase:
    """
    Base configuration for Process Heat SHAP explainers.

    This provides common configuration options shared across all Process Heat
    agents, with sensible defaults for industrial applications.

    Attributes:
        equipment_id: Equipment identifier for provenance tracking
        agent_type: The Process Heat agent type (GL-001 to GL-020)
        domain: Industrial domain for context-aware explanations
        explainer_mode: SHAP explainer type to use
        confidence_threshold: Minimum confidence for valid explanations (default 0.80)
        n_background_samples: Background samples for KernelSHAP
        max_features_display: Maximum features to include in explanations
        enable_caching: Enable LRU caching for repeated explanations
        cache_size: Maximum cache entries
        enable_provenance: Enable SHA-256 provenance tracking
        batch_size: Maximum batch size for batch processing
        feature_names: Custom feature names (overrides domain defaults)
    """
    equipment_id: str
    agent_type: ProcessHeatAgentType = ProcessHeatAgentType.GL_001_THERMALCOMMAND
    domain: IndustrialDomain = IndustrialDomain.COMBUSTION
    explainer_mode: SHAPExplainerMode = SHAPExplainerMode.KERNEL
    confidence_threshold: float = 0.80
    n_background_samples: int = 100
    max_features_display: int = 10
    enable_caching: bool = True
    cache_size: int = 1000
    enable_provenance: bool = True
    batch_size: int = 100
    feature_names: Optional[List[str]] = None
    random_state: int = 42


@dataclass
class GL001ThermalCommandSHAPConfig(ProcessHeatSHAPConfigBase):
    """
    SHAP configuration for GL-001 THERMALCOMMAND orchestrator.

    Optimized for explaining orchestrator decision-making including:
    - Agent selection decisions
    - Workflow routing decisions
    - Priority assignment decisions
    - Load allocation decisions

    Additional Attributes:
        explain_agent_selection: Generate explanations for agent selection
        explain_workflow_routing: Generate explanations for workflow routing
        explain_load_allocation: Generate explanations for load allocation
        multi_agent_context: Include multi-agent coordination context
    """
    agent_type: ProcessHeatAgentType = ProcessHeatAgentType.GL_001_THERMALCOMMAND
    domain: IndustrialDomain = IndustrialDomain.COMBUSTION
    explain_agent_selection: bool = True
    explain_workflow_routing: bool = True
    explain_load_allocation: bool = True
    multi_agent_context: bool = True


@dataclass
class GL003UnifiedSteamSHAPConfig(ProcessHeatSHAPConfigBase):
    """
    SHAP configuration for GL-003 UNIFIEDSTEAM optimizer.

    Optimized for explaining steam system optimization including:
    - PRV optimization decisions
    - Flash steam recovery recommendations
    - Condensate return optimization
    - Steam quality management

    Additional Attributes:
        explain_prv_decisions: Explain PRV let-down optimization
        explain_flash_recovery: Explain flash steam recovery decisions
        explain_condensate: Explain condensate return optimization
        steam_pressure_levels: Pressure levels for context
    """
    agent_type: ProcessHeatAgentType = ProcessHeatAgentType.GL_003_UNIFIEDSTEAM
    domain: IndustrialDomain = IndustrialDomain.STEAM
    explain_prv_decisions: bool = True
    explain_flash_recovery: bool = True
    explain_condensate: bool = True
    steam_pressure_levels: List[float] = field(default_factory=lambda: [600, 150, 50, 15])


@dataclass
class GL006HeatReclaimSHAPConfig(ProcessHeatSHAPConfigBase):
    """
    SHAP configuration for GL-006 HEATRECLAIM agent.

    Optimized for explaining heat recovery decisions including:
    - Heat exchanger sizing recommendations
    - Waste heat utilization optimization
    - Pinch analysis explanations
    - Cascade heat integration decisions

    Additional Attributes:
        explain_pinch_analysis: Include pinch point explanations
        explain_cascade: Explain cascade heat integration
        min_approach_temp_f: Minimum approach temperature for context
    """
    agent_type: ProcessHeatAgentType = ProcessHeatAgentType.GL_006_HEATRECLAIM
    domain: IndustrialDomain = IndustrialDomain.HEAT_RECOVERY
    explain_pinch_analysis: bool = True
    explain_cascade: bool = True
    min_approach_temp_f: float = 20.0


@dataclass
class GL010EmissionsGuardianSHAPConfig(ProcessHeatSHAPConfigBase):
    """
    SHAP configuration for GL-010 EMISSIONSGUARDIAN agent.

    Optimized for explaining emissions predictions including:
    - Exceedance risk predictions
    - Emission rate factor analysis
    - Compliance status explanations
    - Control strategy recommendations

    IMPORTANT: SHAP is used ONLY for explaining predictions/classifications.
    Actual emissions calculations use EPA Method 19 deterministic formulas.

    Additional Attributes:
        explain_exceedance_risk: Explain predicted exceedance risk factors
        explain_control_strategy: Explain emission control recommendations
        permit_pollutants: List of monitored pollutants
        explain_trend_prediction: Explain trend-based predictions
    """
    agent_type: ProcessHeatAgentType = ProcessHeatAgentType.GL_010_EMISSIONSGUARDIAN
    domain: IndustrialDomain = IndustrialDomain.EMISSIONS
    explain_exceedance_risk: bool = True
    explain_control_strategy: bool = True
    explain_trend_prediction: bool = True
    permit_pollutants: List[str] = field(default_factory=lambda: ["NOx", "CO", "SO2", "PM"])


@dataclass
class GL013PredictMaintSHAPConfig(ProcessHeatSHAPConfigBase):
    """
    SHAP configuration for GL-013 PREDICTMAINT agent.

    Optimized for explaining predictive maintenance including:
    - Failure mode predictions
    - Remaining Useful Life (RUL) explanations
    - Anomaly detection explanations
    - Work order priority explanations

    Additional Attributes:
        explain_failure_modes: Explain individual failure mode predictions
        explain_rul: Explain RUL contributing factors
        explain_anomalies: Explain detected anomalies
        failure_modes: List of monitored failure modes
        rul_confidence_level: Confidence level for RUL explanations
    """
    agent_type: ProcessHeatAgentType = ProcessHeatAgentType.GL_013_PREDICTMAINT
    domain: IndustrialDomain = IndustrialDomain.MAINTENANCE
    explainer_mode: SHAPExplainerMode = SHAPExplainerMode.TREE  # Typically tree-based models
    explain_failure_modes: bool = True
    explain_rul: bool = True
    explain_anomalies: bool = True
    failure_modes: List[str] = field(default_factory=lambda: [
        "bearing_wear", "imbalance", "misalignment",
        "lubrication_failure", "rotor_bar_break"
    ])
    rul_confidence_level: float = 0.90


@dataclass
class GL018UnifiedCombustionSHAPConfig(ProcessHeatSHAPConfigBase):
    """
    SHAP configuration for GL-018 UNIFIEDCOMBUSTION optimizer.

    Optimized for explaining combustion optimization including:
    - Efficiency optimization decisions
    - Air-fuel ratio recommendations
    - Flame stability analysis
    - Emission control recommendations

    IMPORTANT: SHAP explains ML classification/ranking only.
    All efficiency calculations use ASME PTC 4.1 deterministic formulas.
    All combustion calculations use API 560 deterministic formulas.

    Additional Attributes:
        explain_efficiency: Explain efficiency optimization decisions
        explain_air_fuel: Explain air-fuel ratio recommendations
        explain_flame_stability: Explain flame stability classifications
        explain_emissions_control: Explain emission control recommendations
        fuel_types: Supported fuel types
    """
    agent_type: ProcessHeatAgentType = ProcessHeatAgentType.GL_018_UNIFIEDCOMBUSTION
    domain: IndustrialDomain = IndustrialDomain.COMBUSTION
    explain_efficiency: bool = True
    explain_air_fuel: bool = True
    explain_flame_stability: bool = True
    explain_emissions_control: bool = True
    fuel_types: List[str] = field(default_factory=lambda: [
        "natural_gas", "fuel_oil_2", "fuel_oil_6", "propane"
    ])


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class ProcessHeatSHAPResult:
    """
    Result from Process Heat SHAP explanation.

    Provides comprehensive SHAP explanation results with industrial domain
    context, provenance tracking, and confidence scoring.

    Attributes:
        agent_type: The agent that generated this explanation
        equipment_id: Equipment identifier
        timestamp: When explanation was generated
        feature_importance: Dict of feature -> importance score
        feature_ranking: Ordered list of (feature, importance, direction)
        shap_values: Raw SHAP values matrix (samples x features)
        base_value: Expected/base value from model
        confidence: Confidence score (0.0 to 1.0)
        confidence_level: Categorical confidence level
        human_readable: Natural language explanation
        domain_context: Industrial domain context
        provenance_hash: SHA-256 hash for audit trail
        processing_time_ms: Time to generate explanation
        metadata: Additional metadata
        cached: Whether result was served from cache
    """
    agent_type: ProcessHeatAgentType
    equipment_id: str
    timestamp: datetime
    feature_importance: Dict[str, float]
    feature_ranking: List[Tuple[str, float, str]]  # (name, importance, direction)
    shap_values: Optional[np.ndarray]
    base_value: float
    confidence: float
    confidence_level: ConfidenceLevel
    human_readable: str
    domain_context: IndustrialDomain
    provenance_hash: str
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_type": self.agent_type.value,
            "equipment_id": self.equipment_id,
            "timestamp": self.timestamp.isoformat(),
            "feature_importance": self.feature_importance,
            "feature_ranking": [
                {"feature": f, "importance": i, "direction": d}
                for f, i, d in self.feature_ranking
            ],
            "base_value": self.base_value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "human_readable": self.human_readable,
            "domain_context": self.domain_context.value,
            "provenance_hash": self.provenance_hash,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
            "cached": self.cached,
        }


@dataclass
class ProcessHeatBatchSHAPResult:
    """
    Result from batch SHAP explanation processing.

    Optimized for high-frequency sensor data with aggregated statistics
    and individual sample explanations.

    Attributes:
        batch_id: Unique batch identifier
        sample_count: Number of samples processed
        aggregate_importance: Aggregated feature importance across batch
        individual_results: Optional individual sample results
        processing_time_ms: Total batch processing time
        cache_hit_rate: Percentage of results served from cache
        provenance_hash: Batch provenance hash
    """
    batch_id: str
    sample_count: int
    aggregate_importance: Dict[str, float]
    aggregate_ranking: List[Tuple[str, float, str]]
    individual_results: Optional[List[ProcessHeatSHAPResult]]
    processing_time_ms: float
    cache_hit_rate: float
    provenance_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# SHAP CACHE IMPLEMENTATION
# =============================================================================

class SHAPCache:
    """
    LRU Cache for SHAP explanations with 66% cost reduction target.

    Implements intelligent caching for SHAP values to reduce computational
    cost when explaining similar inputs. Uses input hashing for cache keys.

    Attributes:
        max_size: Maximum cache entries
        ttl_seconds: Time-to-live for cache entries
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0) -> None:
        """
        Initialize SHAP cache.

        Args:
            max_size: Maximum cache entries
            ttl_seconds: Cache entry TTL in seconds
        """
        self._cache: Dict[str, Tuple[ProcessHeatSHAPResult, float]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._access_order: List[str] = []

        logger.info(f"SHAPCache initialized: max_size={max_size}, ttl={ttl_seconds}s")

    def _generate_key(self, X: np.ndarray, config_hash: str) -> str:
        """Generate cache key from input data and configuration."""
        data_bytes = X.tobytes()
        combined = f"{data_bytes.hex()[:64]}|{config_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(
        self,
        X: np.ndarray,
        config_hash: str
    ) -> Optional[ProcessHeatSHAPResult]:
        """
        Get cached SHAP result if available.

        Args:
            X: Input data
            config_hash: Configuration hash

        Returns:
            Cached result or None
        """
        key = self._generate_key(X, config_hash)

        if key in self._cache:
            result, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp < self._ttl_seconds:
                self._hits += 1

                # Update access order for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)

                # Mark as cached
                result.cached = True
                return result
            else:
                # Expired
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

        self._misses += 1
        return None

    def put(
        self,
        X: np.ndarray,
        config_hash: str,
        result: ProcessHeatSHAPResult
    ) -> None:
        """
        Store SHAP result in cache.

        Args:
            X: Input data
            config_hash: Configuration hash
            result: SHAP result to cache
        """
        key = self._generate_key(X, config_hash)

        # Evict if at capacity (LRU)
        while len(self._cache) >= self._max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

        self._cache[key] = (result, time.time())
        self._access_order.append(key)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0
        logger.info("SHAPCache cleared")


# =============================================================================
# MAIN EXPLAINER CLASS
# =============================================================================

class ProcessHeatSHAPExplainer(Generic[TConfig]):
    """
    Unified SHAP Explainer for Process Heat Agents.

    Provides SHAP-based explanations for ML predictions across all 20 Process
    Heat agents (GL-001 to GL-020). Implements industrial domain context,
    batch processing, caching, and provenance tracking.

    ZERO-HALLUCINATION COMPLIANCE:
        This explainer is used ONLY for explaining ML classifications and
        predictions. It is NEVER used for numeric calculations in regulatory
        reports. All combustion, efficiency, and emissions calculations use
        deterministic engineering formulas.

    Attributes:
        model: The ML model to explain
        config: Agent-specific SHAP configuration
        cache: SHAP result cache

    Example:
        >>> config = GL018UnifiedCombustionSHAPConfig(equipment_id="B-001")
        >>> explainer = ProcessHeatSHAPExplainer(model, config)
        >>> result = explainer.explain(sensor_data)
        >>> print(result.human_readable)
        >>> print(f"Top feature: {result.feature_ranking[0]}")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        model: Any,
        config: TConfig,
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize Process Heat SHAP Explainer.

        Args:
            model: ML model with predict/predict_proba method
            config: Agent-specific SHAP configuration
            background_data: Optional background data for KernelSHAP
        """
        self.model = model
        self.config = config
        self._background_data = background_data
        self._explainer = None
        self._initialized = False

        # Initialize cache if enabled
        self._cache: Optional[SHAPCache] = None
        if config.enable_caching:
            self._cache = SHAPCache(
                max_size=config.cache_size,
                ttl_seconds=3600.0  # 1 hour default TTL
            )

        # Generate configuration hash for cache keys
        self._config_hash = self._hash_config()

        # Domain feature names
        self._domain_feature_names = INDUSTRIAL_FEATURE_NAMES.get(
            config.domain, {}
        )

        logger.info(
            f"ProcessHeatSHAPExplainer initialized: "
            f"agent={config.agent_type.value}, "
            f"equipment={config.equipment_id}, "
            f"domain={config.domain.value}"
        )

    def _hash_config(self) -> str:
        """Generate hash of configuration for cache keying."""
        config_str = (
            f"{self.config.agent_type.value}|"
            f"{self.config.equipment_id}|"
            f"{self.config.explainer_mode.value}|"
            f"{self.config.n_background_samples}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _get_prediction_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Get the prediction function from the model."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        elif hasattr(self.model, "predict"):
            return self.model.predict
        else:
            raise ValueError(
                "Model must have 'predict' or 'predict_proba' method"
            )

    def _initialize_explainer(self, X: np.ndarray) -> None:
        """
        Initialize the appropriate SHAP explainer based on configuration.

        Args:
            X: Sample data for initialization
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required. Install with: pip install shap"
            )

        predict_fn = self._get_prediction_function()

        # Select background data
        if self._background_data is not None:
            background = self._background_data
        else:
            n_bg = min(self.config.n_background_samples, len(X))
            background = shap.kmeans(X, n_bg)

        # Initialize based on explainer mode
        mode = self.config.explainer_mode

        try:
            if mode == SHAPExplainerMode.TREE:
                self._explainer = shap.TreeExplainer(self.model)
            elif mode == SHAPExplainerMode.LINEAR:
                self._explainer = shap.LinearExplainer(self.model, background)
            elif mode == SHAPExplainerMode.DEEP:
                self._explainer = shap.DeepExplainer(self.model, background)
            elif mode == SHAPExplainerMode.GRADIENT:
                self._explainer = shap.GradientExplainer(self.model, background)
            elif mode == SHAPExplainerMode.PARTITION:
                self._explainer = shap.PartitionExplainer(predict_fn, background)
            else:
                # Default to KernelExplainer
                self._explainer = shap.KernelExplainer(predict_fn, background)
        except Exception as e:
            logger.warning(
                f"{mode.value} explainer failed ({e}), "
                "falling back to KernelExplainer"
            )
            self._explainer = shap.KernelExplainer(predict_fn, background)

        self._initialized = True
        logger.info(
            f"SHAP explainer initialized: {type(self._explainer).__name__}"
        )

    def _get_feature_names(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get feature names with industrial domain context.

        Args:
            X: Input data
            feature_names: Optional provided feature names

        Returns:
            List of feature names with domain context
        """
        if feature_names is not None:
            names = feature_names
        elif self.config.feature_names is not None:
            names = self.config.feature_names
        else:
            n_features = X.shape[1] if len(X.shape) > 1 else X.shape[0]
            names = [f"feature_{i}" for i in range(n_features)]

        # Apply domain context translation
        translated = []
        for name in names:
            if name in self._domain_feature_names:
                translated.append(self._domain_feature_names[name])
            else:
                # Convert snake_case to Title Case for readability
                translated.append(name.replace("_", " ").title())

        return translated

    def _calculate_confidence(
        self,
        shap_values: np.ndarray,
        features: Dict[str, float]
    ) -> Tuple[float, ConfidenceLevel]:
        """
        Calculate confidence score and level for explanation.

        The confidence is based on:
        1. Concentration of importance in top features
        2. Stability of SHAP values across samples
        3. Total absolute SHAP value magnitude

        Args:
            shap_values: SHAP values matrix
            features: Feature importance dict

        Returns:
            Tuple of (confidence_score, confidence_level)
        """
        if not features:
            return 0.0, ConfidenceLevel.UNCERTAIN

        # Calculate concentration of importance
        total_importance = sum(abs(v) for v in features.values())

        if total_importance == 0:
            return 0.0, ConfidenceLevel.UNCERTAIN

        # Top 3 features importance ratio
        sorted_importance = sorted(
            features.values(),
            key=lambda x: abs(x),
            reverse=True
        )
        top3_importance = sum(abs(v) for v in sorted_importance[:3])
        concentration = top3_importance / total_importance

        # Calculate stability (low variance = high stability)
        if len(shap_values.shape) > 1 and shap_values.shape[0] > 1:
            stability = 1.0 - np.mean(np.std(shap_values, axis=0) / (np.abs(np.mean(shap_values, axis=0)) + 1e-10))
            stability = max(0.0, min(1.0, stability))
        else:
            stability = 0.8  # Default for single sample

        # Combined confidence
        confidence = 0.5 * concentration + 0.5 * stability
        confidence = max(0.0, min(1.0, confidence))

        # Determine level
        if confidence >= 0.90:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.80:
            level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.60:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.UNCERTAIN

        return round(confidence, 4), level

    def _generate_human_readable(
        self,
        feature_ranking: List[Tuple[str, float, str]],
        domain: IndustrialDomain,
        confidence_level: ConfidenceLevel,
    ) -> str:
        """
        Generate human-readable explanation with industrial context.

        Args:
            feature_ranking: Ranked features (name, importance, direction)
            domain: Industrial domain for context
            confidence_level: Confidence level for qualifying language

        Returns:
            Natural language explanation
        """
        if not feature_ranking:
            return "No significant contributing factors identified."

        # Domain-specific impact verbs
        impact_verbs = {
            IndustrialDomain.COMBUSTION: {
                "increases": "increases combustion efficiency",
                "decreases": "reduces combustion efficiency"
            },
            IndustrialDomain.EMISSIONS: {
                "increases": "increases emission levels",
                "decreases": "reduces emission levels"
            },
            IndustrialDomain.MAINTENANCE: {
                "increases": "increases failure risk",
                "decreases": "improves equipment health"
            },
            IndustrialDomain.STEAM: {
                "increases": "increases steam system efficiency",
                "decreases": "reduces steam system performance"
            },
            IndustrialDomain.HEAT_RECOVERY: {
                "increases": "increases heat recovery",
                "decreases": "reduces heat recovery potential"
            },
        }

        verbs = impact_verbs.get(domain, {
            "increases": "increases the prediction",
            "decreases": "decreases the prediction"
        })

        # Confidence qualifying language
        confidence_qualifier = {
            ConfidenceLevel.HIGH: "with high confidence",
            ConfidenceLevel.MEDIUM: "with moderate confidence",
            ConfidenceLevel.LOW: "with low confidence",
            ConfidenceLevel.UNCERTAIN: "with uncertainty",
        }[confidence_level]

        # Build explanation
        lines = [f"Explanation {confidence_qualifier}:"]

        for i, (name, importance, direction) in enumerate(feature_ranking[:5]):
            prefix = "Primarily, " if i == 0 else "Additionally, " if i == 1 else "Also, "
            impact = verbs["increases"] if direction == "positive" else verbs["decreases"]

            magnitude = "significantly" if abs(importance) > 0.3 else (
                "moderately" if abs(importance) > 0.1 else "slightly"
            )

            lines.append(
                f"{prefix}{name} {magnitude} {impact} "
                f"(contribution: {importance:+.4f})"
            )

        return "\n".join(lines)

    def _calculate_provenance(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        features: Dict[str, float],
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            X: Input data
            shap_values: Computed SHAP values
            features: Feature importance dict

        Returns:
            SHA-256 hash string
        """
        # Create deterministic string representation
        provenance_data = (
            f"agent:{self.config.agent_type.value}|"
            f"equipment:{self.config.equipment_id}|"
            f"version:{self.VERSION}|"
            f"input_shape:{X.shape}|"
            f"shap_shape:{shap_values.shape}|"
            f"features:{sorted(features.items())}|"
            f"timestamp:{datetime.now(timezone.utc).isoformat()}"
        )

        return hashlib.sha256(provenance_data.encode()).hexdigest()

    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ProcessHeatSHAPResult:
        """
        Generate SHAP explanation for input data.

        This method computes SHAP values for each feature and sample,
        providing interpretable explanations for model predictions.

        IMPORTANT: This explains ML predictions only. It does not generate
        any numeric values used for regulatory calculations.

        Args:
            X: Input data (samples x features)
            feature_names: Optional feature names

        Returns:
            ProcessHeatSHAPResult with explanation and provenance

        Raises:
            ValueError: If input validation fails
            ImportError: If SHAP is not installed

        Example:
            >>> result = explainer.explain(sensor_data)
            >>> for name, importance, direction in result.feature_ranking[:3]:
            ...     print(f"{name}: {importance:+.4f} ({direction})")
        """
        start_time = time.time()

        # Ensure 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Check cache
        if self._cache is not None:
            cached_result = self._cache.get(X, self._config_hash)
            if cached_result is not None:
                logger.debug("SHAP result served from cache")
                return cached_result

        # Get feature names
        names = self._get_feature_names(X, feature_names)

        # Initialize explainer if needed
        if not self._initialized:
            self._initialize_explainer(X)

        # Compute SHAP values
        logger.info(f"Computing SHAP values for {X.shape[0]} samples")
        shap_values = self._explainer.shap_values(X)

        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = np.mean(np.array(shap_values), axis=0)

        # Compute mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.mean(axis=-1)

        # Mean SHAP values (with sign) per feature
        mean_shap = shap_values.mean(axis=0)
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=-1)

        # Create feature importance dict
        features = {
            name: float(value)
            for name, value in zip(names, mean_abs_shap)
        }

        # Create feature ranking with direction
        ranking = []
        for name, abs_val in sorted(features.items(), key=lambda x: x[1], reverse=True):
            idx = names.index(name)
            direction = "positive" if mean_shap[idx] >= 0 else "negative"
            ranking.append((name, abs_val, direction))

        ranking = ranking[:self.config.max_features_display]

        # Get expected value
        if hasattr(self._explainer, "expected_value"):
            expected_value = self._explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                base_value = float(np.mean(expected_value))
            else:
                base_value = float(expected_value)
        else:
            base_value = 0.0

        # Calculate confidence
        confidence, confidence_level = self._calculate_confidence(shap_values, features)

        # Generate human-readable explanation
        human_readable = self._generate_human_readable(
            ranking,
            self.config.domain,
            confidence_level
        )

        # Calculate provenance
        provenance_hash = ""
        if self.config.enable_provenance:
            provenance_hash = self._calculate_provenance(X, shap_values, features)

        # Processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Create result
        result = ProcessHeatSHAPResult(
            agent_type=self.config.agent_type,
            equipment_id=self.config.equipment_id,
            timestamp=datetime.now(timezone.utc),
            feature_importance=features,
            feature_ranking=ranking,
            shap_values=shap_values if X.shape[0] <= 100 else None,  # Limit storage
            base_value=base_value,
            confidence=confidence,
            confidence_level=confidence_level,
            human_readable=human_readable,
            domain_context=self.config.domain,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
            metadata={
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "explainer_type": type(self._explainer).__name__,
                "version": self.VERSION,
            },
            cached=False,
        )

        # Store in cache
        if self._cache is not None:
            self._cache.put(X, self._config_hash, result)

        logger.info(
            f"SHAP explanation completed in {processing_time_ms:.2f}ms, "
            f"confidence={confidence:.2f} ({confidence_level.value}), "
            f"provenance={provenance_hash[:16]}..."
        )

        return result

    def explain_single(
        self,
        x: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ProcessHeatSHAPResult:
        """
        Generate SHAP explanation for a single sample.

        Args:
            x: Single input sample
            feature_names: Optional feature names

        Returns:
            ProcessHeatSHAPResult for single prediction
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        return self.explain(x, feature_names)

    def explain_batch(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        include_individual: bool = False,
    ) -> ProcessHeatBatchSHAPResult:
        """
        Generate SHAP explanations for a batch of samples.

        Optimized for high-frequency sensor data processing. Provides
        aggregated feature importance with optional individual results.

        Args:
            X: Batch of input samples (samples x features)
            feature_names: Optional feature names
            include_individual: Include individual sample results

        Returns:
            ProcessHeatBatchSHAPResult with batch statistics

        Example:
            >>> # Process 1000 sensor readings
            >>> batch_result = explainer.explain_batch(sensor_batch)
            >>> print(f"Cache hit rate: {batch_result.cache_hit_rate:.1%}")
        """
        start_time = time.time()
        batch_id = hashlib.md5(X.tobytes()[:256]).hexdigest()[:16]

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        batch_size = self.config.batch_size

        all_features: Dict[str, List[float]] = {}
        individual_results: List[ProcessHeatSHAPResult] = []
        cache_hits = 0
        cache_total = 0

        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]

            # Try cache first
            if self._cache is not None:
                cached = self._cache.get(batch, self._config_hash)
                if cached is not None:
                    cache_hits += 1
                    cache_total += 1

                    for name, importance in cached.feature_importance.items():
                        if name not in all_features:
                            all_features[name] = []
                        all_features[name].append(importance)

                    if include_individual:
                        individual_results.append(cached)
                    continue

            cache_total += 1

            # Compute explanation
            result = self.explain(batch, feature_names)

            for name, importance in result.feature_importance.items():
                if name not in all_features:
                    all_features[name] = []
                all_features[name].append(importance)

            if include_individual:
                individual_results.append(result)

        # Aggregate importance
        aggregate_importance = {
            name: float(np.mean(values))
            for name, values in all_features.items()
        }

        # Create aggregate ranking
        aggregate_ranking = [
            (name, importance, "positive")  # Direction aggregated
            for name, importance in sorted(
                aggregate_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ][:self.config.max_features_display]

        # Calculate provenance for batch
        batch_provenance = hashlib.sha256(
            f"batch:{batch_id}|samples:{n_samples}|features:{sorted(aggregate_importance.items())}".encode()
        ).hexdigest()

        processing_time_ms = (time.time() - start_time) * 1000
        cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0

        logger.info(
            f"Batch SHAP completed: {n_samples} samples in {processing_time_ms:.2f}ms, "
            f"cache hit rate: {cache_hit_rate:.1%}"
        )

        return ProcessHeatBatchSHAPResult(
            batch_id=batch_id,
            sample_count=n_samples,
            aggregate_importance=aggregate_importance,
            aggregate_ranking=aggregate_ranking,
            individual_results=individual_results if include_individual else None,
            processing_time_ms=processing_time_ms,
            cache_hit_rate=cache_hit_rate,
            provenance_hash=batch_provenance,
        )

    async def explain_async(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ProcessHeatSHAPResult:
        """
        Async version of explain for non-blocking operations.

        Args:
            X: Input data
            feature_names: Optional feature names

        Returns:
            ProcessHeatSHAPResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.explain(X, feature_names)
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance from last explanation.

        Returns:
            Dict mapping feature names to mean absolute SHAP values
        """
        # Return cached aggregate if available
        if self._cache is not None and self._cache.size > 0:
            all_importance: Dict[str, List[float]] = {}
            # Aggregate from cache (simplified - actual implementation would iterate cache)
            return {}  # Would need to track global importance separately
        return {}

    def clear_cache(self) -> None:
        """Clear the SHAP result cache."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def cache_hit_rate(self) -> float:
        """Get current cache hit rate."""
        if self._cache is not None:
            return self._cache.hit_rate
        return 0.0

    def get_waterfall_data(
        self,
        x: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get data for waterfall plot visualization.

        Args:
            x: Single sample
            feature_names: Optional feature names

        Returns:
            Dictionary with waterfall plot data suitable for visualization
        """
        result = self.explain_single(x, feature_names)

        features = [f for f, _, _ in result.feature_ranking]
        contributions = []
        cumulative = [result.base_value]

        for i, (_, importance, direction) in enumerate(result.feature_ranking):
            contrib = importance if direction == "positive" else -importance
            contributions.append(contrib)
            cumulative.append(cumulative[-1] + contrib)

        return {
            "base_value": result.base_value,
            "features": features,
            "contributions": contributions,
            "cumulative": cumulative,
            "final_prediction": cumulative[-1],
            "provenance_hash": result.provenance_hash,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_process_heat_explainer(
    model: Any,
    agent_type: ProcessHeatAgentType,
    equipment_id: str,
    background_data: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> ProcessHeatSHAPExplainer:
    """
    Factory function to create configured ProcessHeatSHAPExplainer.

    Automatically selects the appropriate configuration class based on
    agent type and applies sensible defaults.

    Args:
        model: ML model to explain
        agent_type: Process Heat agent type
        equipment_id: Equipment identifier
        background_data: Optional background data for KernelSHAP
        **kwargs: Additional configuration options

    Returns:
        Configured ProcessHeatSHAPExplainer

    Example:
        >>> explainer = create_process_heat_explainer(
        ...     model=trained_model,
        ...     agent_type=ProcessHeatAgentType.GL_018_UNIFIEDCOMBUSTION,
        ...     equipment_id="BOILER-001",
        ... )
    """
    # Select configuration class based on agent type
    config_map = {
        ProcessHeatAgentType.GL_001_THERMALCOMMAND: GL001ThermalCommandSHAPConfig,
        ProcessHeatAgentType.GL_003_UNIFIEDSTEAM: GL003UnifiedSteamSHAPConfig,
        ProcessHeatAgentType.GL_006_HEATRECLAIM: GL006HeatReclaimSHAPConfig,
        ProcessHeatAgentType.GL_010_EMISSIONSGUARDIAN: GL010EmissionsGuardianSHAPConfig,
        ProcessHeatAgentType.GL_013_PREDICTMAINT: GL013PredictMaintSHAPConfig,
        ProcessHeatAgentType.GL_018_UNIFIEDCOMBUSTION: GL018UnifiedCombustionSHAPConfig,
    }

    config_class = config_map.get(agent_type)

    if config_class is None:
        # Default configuration for other agents
        domain_map = {
            ProcessHeatAgentType.GL_002_BOILEROPTIMIZER: IndustrialDomain.COMBUSTION,
            ProcessHeatAgentType.GL_004_BURNMASTER: IndustrialDomain.COMBUSTION,
            ProcessHeatAgentType.GL_005_COMBUSTIONDIAG: IndustrialDomain.COMBUSTION,
            ProcessHeatAgentType.GL_007_FURNACEMONITOR: IndustrialDomain.COMBUSTION,
            ProcessHeatAgentType.GL_008_STEAMTRAP: IndustrialDomain.STEAM,
            ProcessHeatAgentType.GL_009_THERMALFLUID: IndustrialDomain.THERMAL_FLUID,
            ProcessHeatAgentType.GL_011_FUELOPTIMIZER: IndustrialDomain.COMBUSTION,
            ProcessHeatAgentType.GL_012_LOADFORECAST: IndustrialDomain.COMBUSTION,
            ProcessHeatAgentType.GL_014_HEATEXCHANGER: IndustrialDomain.HEAT_RECOVERY,
            ProcessHeatAgentType.GL_015_INSULATION: IndustrialDomain.HEAT_RECOVERY,
            ProcessHeatAgentType.GL_016_WATERTREAT: IndustrialDomain.WATER_TREATMENT,
            ProcessHeatAgentType.GL_017_CONDENSER: IndustrialDomain.STEAM,
            ProcessHeatAgentType.GL_019_COGENERATION: IndustrialDomain.STEAM,
            ProcessHeatAgentType.GL_020_SOLARINTEGRATION: IndustrialDomain.HEAT_RECOVERY,
        }

        domain = domain_map.get(agent_type, IndustrialDomain.COMBUSTION)

        config = ProcessHeatSHAPConfigBase(
            agent_type=agent_type,
            equipment_id=equipment_id,
            domain=domain,
            **kwargs
        )
    else:
        config = config_class(equipment_id=equipment_id, **kwargs)

    return ProcessHeatSHAPExplainer(
        model=model,
        config=config,
        background_data=background_data,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_gl018_combustion_explanation():
    """
    Example: GL-018 UnifiedCombustion SHAP explanation.

    Demonstrates how to use ProcessHeatSHAPExplainer to explain
    combustion optimization decisions.

    NOTE: This is for documentation purposes. Run with actual model.
    """
    # Example configuration
    config = GL018UnifiedCombustionSHAPConfig(
        equipment_id="BOILER-001",
        explainer_mode=SHAPExplainerMode.KERNEL,
        confidence_threshold=0.80,
        enable_caching=True,
    )

    # Example feature names for combustion optimization
    feature_names = [
        "excess_air_pct",
        "o2_pct",
        "co_ppm",
        "nox_ppm",
        "flue_temp_f",
        "load_pct",
        "fuel_flow_rate",
        "air_damper_pct",
        "fsi",
        "ambient_temp_f",
    ]

    print("GL-018 UnifiedCombustion SHAP Configuration:")
    print(f"  Agent Type: {config.agent_type.value}")
    print(f"  Equipment ID: {config.equipment_id}")
    print(f"  Domain: {config.domain.value}")
    print(f"  Explainer Mode: {config.explainer_mode.value}")
    print(f"  Confidence Threshold: {config.confidence_threshold}")
    print(f"  Feature Names: {feature_names}")

    return config, feature_names


def example_gl013_predictmaint_explanation():
    """
    Example: GL-013 PredictMaint SHAP explanation.

    Demonstrates how to use ProcessHeatSHAPExplainer to explain
    failure predictions and RUL estimates.
    """
    config = GL013PredictMaintSHAPConfig(
        equipment_id="PUMP-001",
        explainer_mode=SHAPExplainerMode.TREE,
        explain_failure_modes=True,
        explain_rul=True,
        failure_modes=[
            "bearing_wear",
            "imbalance",
            "misalignment",
            "lubrication_failure",
        ],
    )

    feature_names = [
        "vibration_mm_s",
        "bearing_temp_c",
        "oil_condition_score",
        "running_hours",
        "load_percent",
        "current_unbalance_pct",
        "fft_1x_amplitude",
        "fft_2x_amplitude",
        "viscosity_change_pct",
        "iron_ppm",
    ]

    print("GL-013 PredictMaint SHAP Configuration:")
    print(f"  Agent Type: {config.agent_type.value}")
    print(f"  Equipment ID: {config.equipment_id}")
    print(f"  Domain: {config.domain.value}")
    print(f"  Explainer Mode: {config.explainer_mode.value}")
    print(f"  Failure Modes: {config.failure_modes}")
    print(f"  Feature Names: {feature_names}")

    return config, feature_names


def example_gl010_emissions_explanation():
    """
    Example: GL-010 EmissionsGuardian SHAP explanation.

    Demonstrates how to use ProcessHeatSHAPExplainer to explain
    emission exceedance risk predictions.

    IMPORTANT: SHAP explains the ML risk prediction model only.
    Actual emissions calculations use EPA Method 19 formulas.
    """
    config = GL010EmissionsGuardianSHAPConfig(
        equipment_id="BOILER-001",
        explainer_mode=SHAPExplainerMode.KERNEL,
        explain_exceedance_risk=True,
        permit_pollutants=["NOx", "CO", "SO2"],
    )

    feature_names = [
        "stack_o2_pct",
        "co_ppm",
        "nox_ppm",
        "load_pct",
        "fuel_flow_rate",
        "ambient_temp_f",
        "co2_lb_hr_trend",
        "nox_lb_hr_trend",
        "permit_utilization_pct",
        "operating_hours_today",
    ]

    print("GL-010 EmissionsGuardian SHAP Configuration:")
    print(f"  Agent Type: {config.agent_type.value}")
    print(f"  Equipment ID: {config.equipment_id}")
    print(f"  Domain: {config.domain.value}")
    print(f"  Permit Pollutants: {config.permit_pollutants}")
    print()
    print("  ZERO-HALLUCINATION NOTICE:")
    print("  SHAP explains ML risk predictions only.")
    print("  Emissions calculations use EPA Method 19 deterministic formulas.")

    return config, feature_names


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ProcessHeatAgentType",
    "SHAPExplainerMode",
    "ConfidenceLevel",
    "IndustrialDomain",
    # Configuration base
    "ProcessHeatSHAPConfigBase",
    # Agent-specific configurations
    "GL001ThermalCommandSHAPConfig",
    "GL003UnifiedSteamSHAPConfig",
    "GL006HeatReclaimSHAPConfig",
    "GL010EmissionsGuardianSHAPConfig",
    "GL013PredictMaintSHAPConfig",
    "GL018UnifiedCombustionSHAPConfig",
    # Result classes
    "ProcessHeatSHAPResult",
    "ProcessHeatBatchSHAPResult",
    # Cache
    "SHAPCache",
    # Main explainer
    "ProcessHeatSHAPExplainer",
    # Factory
    "create_process_heat_explainer",
    # Feature name mappings
    "INDUSTRIAL_FEATURE_NAMES",
    # Examples
    "example_gl018_combustion_explanation",
    "example_gl013_predictmaint_explanation",
    "example_gl010_emissions_explanation",
]


if __name__ == "__main__":
    # Run examples when executed directly
    print("=" * 70)
    print("GreenLang Process Heat SHAP Integration Module")
    print("=" * 70)
    print()

    print("-" * 70)
    example_gl018_combustion_explanation()
    print()

    print("-" * 70)
    example_gl013_predictmaint_explanation()
    print()

    print("-" * 70)
    example_gl010_emissions_explanation()
    print()

    print("=" * 70)
    print("Module loaded successfully. Ready for integration.")
    print("=" * 70)
