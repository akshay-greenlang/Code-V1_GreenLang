"""
GL-015 INSULSCAN Calculator Modules

Zero-hallucination calculation engines for thermal insulation
assessment and building thermography analysis.

All calculations are deterministic with complete provenance
tracking for regulatory compliance.

Modules:
    thermal_image_analyzer: Comprehensive thermal image analysis
    energy_loss_quantifier: Energy waste quantification and carbon footprint
    economic_calculator: Economic impact analysis for insulation decisions
        - Repair cost estimation (materials, labor, access, permits)
        - Energy savings calculation (heat loss, fuel costs)
        - Payback analysis (simple and discounted)
        - NPV and IRR calculation
        - Life cycle cost analysis (TCO)
        - Economic thickness optimization (ASTM C680)
        - Budget impact analysis (CapEx, OpEx)
        - Carbon economics (credits, taxes, ESG)
    performance_tracker: Historical tracking and trend analysis
        - Performance metrics tracking over time
        - Degradation rate analysis (linear, exponential, Weibull)
        - Remaining useful life (RUL) estimation
        - Inspection history management
        - Trend analysis with anomaly detection
        - Fleet/facility benchmarking
        - Predictive analytics and forecasting
        - KPI dashboard generation
    surface_temperature_analyzer: Environmental corrections and surface analysis
        - Ambient condition normalization (20C reference)
        - Wind speed corrections for convection
        - Solar loading corrections
        - Reflected temperature compensation
        - Personnel protection limit checks (ASTM C1055)
        - Temperature distribution analysis
        - Measurement uncertainty (GUM method)
        - Inspection conditions recommendation
        - Seasonal adjustments
    repair_prioritization: Intelligent repair prioritization engine
        - Multi-factor criticality scoring (heat loss, safety, process, environmental, CUI)
        - ROI-based ranking with NPV analysis
        - Risk-based prioritization (FMEA methodology)
        - Schedule optimization with resource leveling
        - Budget-constrained optimization (knapsack problem)
        - Work order generation with material/labor estimates
"""

# Import from constants module
from .constants import (
    # Physical constants
    PhysicalConstants,
    AirPropertiesTable,
    # Insulation materials
    InsulationType as ConstantsInsulationType,
    InsulationMaterialSpec,
    InsulationMaterials,
    # Surface properties
    EmissivitySpec,
    SurfaceEmissivity,
    # Convection
    ConvectionCorrelation,
    ConvectionCorrelations,
    # Economic (from constants)
    FuelPrice,
    EconomicParameters as ConstantsEconomicParameters,
    # Safety
    SafetyLimit,
    SafetyLimits,
    # Pipe dimensions
    PipeDimension,
    PipeDimensions,
    # Thicknesses
    StandardThicknesses,
    # Operating conditions
    OperatingCondition,
    OperatingConditions,
)

# Import from units module
from .units import (
    # Constants
    ConversionFactors,
    # Temperature
    TemperatureConverter,
    # Thermal conductivity
    ThermalConductivityConverter,
    # R-value
    RValueConverter,
    # Heat loss
    HeatLossConverter,
    # Energy
    EnergyConverter,
    # Power
    PowerConverter,
    # Length and Area
    LengthConverter,
    AreaConverter,
    # Universal
    UniversalConverter,
    # Convenience functions
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    k_si_to_imperial,
    k_imperial_to_si,
    r_si_to_imperial,
    r_imperial_to_si,
)

# Import from provenance module
from .provenance import (
    # Enums
    CalculationType,
    HashAlgorithm,
    RecordStatus,
    # Data classes
    CalculationMetadata,
    CalculationRecord,
    HashChainEntry,
    # Main classes
    ProvenanceTracker,
    BatchProvenanceTracker,
    ProvenanceValidator,
    # Utilities
    CanonicalJsonSerializer,
    compute_hash,
    create_input_hash,
    create_output_hash,
    verify_provenance,
)

from .thermal_image_analyzer import (
    # Enums
    IRCameraResolution,
    ThermalAnomalyType,
    ROIShape,
    ImageQualityLevel,
    # Data structures
    TemperatureMatrixConfig,
    TemperaturePoint,
    TemperatureStatistics,
    Hotspot,
    TemperatureGradient,
    IsothermalContour,
    ROIDefinition,
    ROIAnalysisResult,
    ThermalAnomalyClassification,
    ImageQualityAssessment,
    TemperatureMapResult,
    ThermalAnalysisResult,
    AnomalyThresholds,
    ANOMALY_THRESHOLDS,
    # Main class
    ThermalImageAnalyzer,
    # Utility functions
    create_roi_rectangle,
    create_roi_circle,
    create_roi_polygon,
    verify_analysis_provenance,
)

from .energy_loss_quantifier import (
    # Main Calculator
    EnergyLossQuantifier,
    # Data Models
    InspectionLocation,
    SeasonalProfile,
    FuelConsumption,
    CarbonFootprint,
    BenchmarkComparison,
    SavingsPotential,
    EnergyEfficiencyMetrics,
    AnnualEnergyLoss,
    FacilityAggregation,
    EnergyLossReport,
    # Enums
    FuelType,
    # Constants
    EPA_EMISSION_FACTORS,
    DEFAULT_BOILER_EFFICIENCIES,
    BENCHMARK_HEAT_LOSS_RATES,
    CARBON_PRICE_SCENARIOS,
)

from .economic_calculator import (
    # Main Calculator
    EconomicCalculator,
    # Enumerations
    InsulationType,
    RepairComplexity,
    AccessRequirement,
    EquipmentType,
    DepreciationMethod,
    CarbonPricingScheme,
    # Input Models
    RepairCostInput,
    EnergySavingsInput,
    FinancialParameters,
    LifecycleInput,
    CarbonEconomicsInput,
    EconomicThicknessInput,
    # Result Models
    RepairCostResult,
    EnergySavingsResult,
    PaybackResult,
    NPVIRRResult,
    LifecycleCostResult,
    EconomicThicknessResult,
    BudgetImpactResult,
    CarbonEconomicsResult,
    # Cost Factor Databases
    InsulationMaterialCosts,
    LaborRates,
    AccessCosts,
    ThermalConductivity,
    FuelProperties,
)

from .performance_tracker import (
    # Main class
    InsulationPerformanceTracker,
    # Enumerations
    DegradationModel,
    ConditionCategory,
    AlertSeverity,
    TrendDirection,
    InspectionType,
    # Data models
    PerformanceDataPoint,
    DegradationAnalysis,
    RemainingUsefulLife,
    InspectionRecord,
    TrendAnalysisResult,
    FleetBenchmark,
    FutureConditionForecast,
    KPIDashboard,
    CalculationStep,
)

from .surface_temperature_analyzer import (
    # Main class
    SurfaceTemperatureAnalyzer,
    # Enums
    LocationType,
    SurfaceOrientation,
    SurfaceColor,
    CloudCover,
    Season,
    InspectionCondition,
    # Constants
    STEFAN_BOLTZMANN,
    REFERENCE_AMBIENT_C,
    KELVIN_OFFSET,
    PERSONNEL_PROTECTION_LIMIT_C,
    PERSONNEL_PROTECTION_LIMIT_F,
    SOLAR_ABSORPTIVITY,
    WIND_CORRECTION_FACTORS,
    CLOUD_COVER_FACTORS,
    SKY_TEMPERATURE_DEPRESSION,
    SEASONAL_BASELINE_C,
    # Result data classes
    AmbientNormalizationResult,
    WindCorrectionResult,
    SolarCorrectionResult,
    ReflectedTemperatureResult,
    PersonnelProtectionResult,
    TemperatureDistributionResult,
    MeasurementUncertaintyResult,
    InspectionConditionsResult,
    SeasonalAdjustmentResult,
)

from .repair_prioritization import (
    # Main engine
    RepairPrioritizationEngine,
    # Enumerations
    PriorityCategory,
    RiskLevel,
    InsulationMaterial as RepairInsulationMaterial,
    DamageType,
    RepairScope,
    EquipmentType as RepairEquipmentType,
    OutageType,
    SeverityRating,
    OccurrenceRating,
    DetectionRating,
    # Input models
    DefectLocation,
    ThermalDefect,
    CriticalityWeights,
    EconomicParameters,
    ScheduleConstraints,
    BudgetConstraint,
    # Output models
    CriticalityScore,
    RepairROI,
    RiskAssessment,
    ScheduledRepair,
    WorkScope,
    OptimizedRepairPlan,
    # Validation
    validate_defect_input,
    validate_criticality_weights,
    # Constants
    HEAT_LOSS_SEVERITY_THRESHOLDS,
    PERSONNEL_SAFETY_TEMP_C,
    INSULATION_COST_PER_METER,
    LABOR_RATES as REPAIR_LABOR_RATES,
    PRODUCTION_RATES,
    CUI_RISK_FACTORS,
)

# Import from heat_loss_calculator module (advanced features)
from .heat_loss_calculator import (
    # Advanced Calculator
    AdvancedHeatLossCalculator,
    # Advanced Result classes
    CriticalRadiusResult,
    EconomicThicknessResult,
    ThreeKMethodResult,
    MultiLayerResult,
    InsulationLayer,
)

# Import from thermal_conductivity_library module
from .thermal_conductivity_library import (
    # Enumerations
    InsulationCategory,
    InsulationMaterialType,
    AgingCondition,
    MoistureCondition,
    # Data classes
    ThermalConductivitySpec,
    AgingCorrectionFactor,
    MoistureCorrectionFactor,
    ThermalConductivityResult,
    # Main class
    ThermalConductivityLibrary,
    # Databases
    THERMAL_CONDUCTIVITY_DATABASE,
    AGING_CORRECTION_FACTORS,
    MOISTURE_CORRECTION_FACTORS,
    # Convenience functions
    get_k_value,
    list_available_materials,
)

__all__ = [
    # === Constants Module ===
    # Physical constants
    "PhysicalConstants",
    "AirPropertiesTable",
    # Insulation materials
    "ConstantsInsulationType",
    "InsulationMaterialSpec",
    "InsulationMaterials",
    # Surface properties
    "EmissivitySpec",
    "SurfaceEmissivity",
    # Convection
    "ConvectionCorrelation",
    "ConvectionCorrelations",
    # Economic (from constants)
    "FuelPrice",
    "ConstantsEconomicParameters",
    # Safety
    "SafetyLimit",
    "SafetyLimits",
    # Pipe dimensions
    "PipeDimension",
    "PipeDimensions",
    # Thicknesses
    "StandardThicknesses",
    # Operating conditions
    "OperatingCondition",
    "OperatingConditions",
    # === Units Module ===
    # Conversion factors
    "ConversionFactors",
    # Converters
    "TemperatureConverter",
    "ThermalConductivityConverter",
    "RValueConverter",
    "HeatLossConverter",
    "EnergyConverter",
    "PowerConverter",
    "LengthConverter",
    "AreaConverter",
    "UniversalConverter",
    # Convenience functions
    "celsius_to_fahrenheit",
    "fahrenheit_to_celsius",
    "k_si_to_imperial",
    "k_imperial_to_si",
    "r_si_to_imperial",
    "r_imperial_to_si",
    # === Provenance Module ===
    # Enums
    "CalculationType",
    "HashAlgorithm",
    "RecordStatus",
    # Data classes
    "CalculationMetadata",
    "CalculationRecord",
    "HashChainEntry",
    # Main classes
    "ProvenanceTracker",
    "BatchProvenanceTracker",
    "ProvenanceValidator",
    # Utilities
    "CanonicalJsonSerializer",
    "compute_hash",
    "create_input_hash",
    "create_output_hash",
    "verify_provenance",
    # === Thermal Image Analyzer ===
    # Enums
    "IRCameraResolution",
    "ThermalAnomalyType",
    "ROIShape",
    "ImageQualityLevel",
    # Data structures
    "TemperatureMatrixConfig",
    "TemperaturePoint",
    "TemperatureStatistics",
    "Hotspot",
    "TemperatureGradient",
    "IsothermalContour",
    "ROIDefinition",
    "ROIAnalysisResult",
    "ThermalAnomalyClassification",
    "ImageQualityAssessment",
    "TemperatureMapResult",
    "ThermalAnalysisResult",
    "AnomalyThresholds",
    "ANOMALY_THRESHOLDS",
    # Main class
    "ThermalImageAnalyzer",
    # Utility functions
    "create_roi_rectangle",
    "create_roi_circle",
    "create_roi_polygon",
    "verify_analysis_provenance",
    # === Energy Loss Quantifier ===
    # Main Calculator
    "EnergyLossQuantifier",
    # Data Models
    "InspectionLocation",
    "SeasonalProfile",
    "FuelConsumption",
    "CarbonFootprint",
    "BenchmarkComparison",
    "SavingsPotential",
    "EnergyEfficiencyMetrics",
    "AnnualEnergyLoss",
    "FacilityAggregation",
    "EnergyLossReport",
    # Enums (from energy_loss_quantifier)
    "FuelType",
    # Constants
    "EPA_EMISSION_FACTORS",
    "DEFAULT_BOILER_EFFICIENCIES",
    "BENCHMARK_HEAT_LOSS_RATES",
    "CARBON_PRICE_SCENARIOS",
    # === Economic Calculator ===
    # Main Calculator
    "EconomicCalculator",
    # Enumerations
    "InsulationType",
    "RepairComplexity",
    "AccessRequirement",
    "EquipmentType",
    "DepreciationMethod",
    "CarbonPricingScheme",
    # Input Models
    "RepairCostInput",
    "EnergySavingsInput",
    "FinancialParameters",
    "LifecycleInput",
    "CarbonEconomicsInput",
    "EconomicThicknessInput",
    # Result Models
    "RepairCostResult",
    "EnergySavingsResult",
    "PaybackResult",
    "NPVIRRResult",
    "LifecycleCostResult",
    "EconomicThicknessResult",
    "BudgetImpactResult",
    "CarbonEconomicsResult",
    # Cost Factor Databases
    "InsulationMaterialCosts",
    "LaborRates",
    "AccessCosts",
    "ThermalConductivity",
    "FuelProperties",
    # === Performance Tracker ===
    # Main class
    "InsulationPerformanceTracker",
    # Enumerations
    "DegradationModel",
    "ConditionCategory",
    "AlertSeverity",
    "TrendDirection",
    "InspectionType",
    # Data models
    "PerformanceDataPoint",
    "DegradationAnalysis",
    "RemainingUsefulLife",
    "InspectionRecord",
    "TrendAnalysisResult",
    "FleetBenchmark",
    "FutureConditionForecast",
    "KPIDashboard",
    "CalculationStep",
    # === Surface Temperature Analyzer ===
    # Main class
    "SurfaceTemperatureAnalyzer",
    # Enums
    "LocationType",
    "SurfaceOrientation",
    "SurfaceColor",
    "CloudCover",
    "Season",
    "InspectionCondition",
    # Constants
    "STEFAN_BOLTZMANN",
    "REFERENCE_AMBIENT_C",
    "KELVIN_OFFSET",
    "PERSONNEL_PROTECTION_LIMIT_C",
    "PERSONNEL_PROTECTION_LIMIT_F",
    "SOLAR_ABSORPTIVITY",
    "WIND_CORRECTION_FACTORS",
    "CLOUD_COVER_FACTORS",
    "SKY_TEMPERATURE_DEPRESSION",
    "SEASONAL_BASELINE_C",
    # Result data classes
    "AmbientNormalizationResult",
    "WindCorrectionResult",
    "SolarCorrectionResult",
    "ReflectedTemperatureResult",
    "PersonnelProtectionResult",
    "TemperatureDistributionResult",
    "MeasurementUncertaintyResult",
    "InspectionConditionsResult",
    "SeasonalAdjustmentResult",
    # === Repair Prioritization Engine ===
    # Main engine
    "RepairPrioritizationEngine",
    # Enumerations
    "PriorityCategory",
    "RiskLevel",
    "RepairInsulationMaterial",
    "DamageType",
    "RepairScope",
    "RepairEquipmentType",
    "OutageType",
    "SeverityRating",
    "OccurrenceRating",
    "DetectionRating",
    # Input models
    "DefectLocation",
    "ThermalDefect",
    "CriticalityWeights",
    "EconomicParameters",
    "ScheduleConstraints",
    "BudgetConstraint",
    # Output models
    "CriticalityScore",
    "RepairROI",
    "RiskAssessment",
    "ScheduledRepair",
    "WorkScope",
    "OptimizedRepairPlan",
    # Validation
    "validate_defect_input",
    "validate_criticality_weights",
    # Constants
    "HEAT_LOSS_SEVERITY_THRESHOLDS",
    "PERSONNEL_SAFETY_TEMP_C",
    "INSULATION_COST_PER_METER",
    "REPAIR_LABOR_RATES",
    "PRODUCTION_RATES",
    "CUI_RISK_FACTORS",
    # === Advanced Heat Loss Calculator ===
    "AdvancedHeatLossCalculator",
    "CriticalRadiusResult",
    "EconomicThicknessResult",
    "ThreeKMethodResult",
    "MultiLayerResult",
    "InsulationLayer",
    # === Thermal Conductivity Library ===
    "ThermalConductivityLibrary",
    "InsulationCategory",
    "InsulationMaterialType",
    "AgingCondition",
    "MoistureCondition",
    "ThermalConductivitySpec",
    "AgingCorrectionFactor",
    "MoistureCorrectionFactor",
    "ThermalConductivityResult",
    "THERMAL_CONDUCTIVITY_DATABASE",
    "AGING_CORRECTION_FACTORS",
    "MOISTURE_CORRECTION_FACTORS",
    "get_k_value",
    "list_available_materials",
]

__version__ = "1.0.0"
