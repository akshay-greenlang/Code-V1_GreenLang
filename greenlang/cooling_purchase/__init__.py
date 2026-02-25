"""
AGENT-MRV-012: Cooling Purchase Agent (GL-MRV-X-023)

Scope 2 GHG emissions from purchased district cooling and cooling services
per GHG Protocol Scope 2 Guidance (2015).

18 cooling technologies, 12 regional EFs, 11 heat source EFs, 11 refrigerants.
7-engine architecture with SHA-256 provenance and 84 compliance requirements.
"""

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-X-023"
__agent_label__ = "AGENT-MRV-012"

_LAZY_IMPORTS = {
    "CoolingTechnology": "greenlang.cooling_purchase.models",
    "CompressorType": "greenlang.cooling_purchase.models",
    "CondenserType": "greenlang.cooling_purchase.models",
    "AbsorptionType": "greenlang.cooling_purchase.models",
    "FreeCoolingSource": "greenlang.cooling_purchase.models",
    "TESType": "greenlang.cooling_purchase.models",
    "HeatSource": "greenlang.cooling_purchase.models",
    "EfficiencyMetric": "greenlang.cooling_purchase.models",
    "CoolingUnit": "greenlang.cooling_purchase.models",
    "EmissionGas": "greenlang.cooling_purchase.models",
    "GWPSource": "greenlang.cooling_purchase.models",
    "ComplianceStatus": "greenlang.cooling_purchase.models",
    "DataQualityTier": "greenlang.cooling_purchase.models",
    "FacilityType": "greenlang.cooling_purchase.models",
    "ReportingPeriod": "greenlang.cooling_purchase.models",
    "AggregationType": "greenlang.cooling_purchase.models",
    "BatchStatus": "greenlang.cooling_purchase.models",
    "Refrigerant": "greenlang.cooling_purchase.models",
    "ElectricChillerRequest": "greenlang.cooling_purchase.models",
    "AbsorptionCoolingRequest": "greenlang.cooling_purchase.models",
    "FreeCoolingRequest": "greenlang.cooling_purchase.models",
    "TESRequest": "greenlang.cooling_purchase.models",
    "DistrictCoolingRequest": "greenlang.cooling_purchase.models",
    "CalculationResult": "greenlang.cooling_purchase.models",
    "TESCalculationResult": "greenlang.cooling_purchase.models",
    "BatchCalculationRequest": "greenlang.cooling_purchase.models",
    "BatchCalculationResult": "greenlang.cooling_purchase.models",
    "UncertaintyRequest": "greenlang.cooling_purchase.models",
    "UncertaintyResult": "greenlang.cooling_purchase.models",
    "ComplianceCheckResult": "greenlang.cooling_purchase.models",
    "AggregationRequest": "greenlang.cooling_purchase.models",
    "AggregationResult": "greenlang.cooling_purchase.models",
    "CoolingPurchaseConfig": "greenlang.cooling_purchase.config",
    "CoolingPurchaseMetrics": "greenlang.cooling_purchase.metrics",
    "CoolingPurchaseProvenance": "greenlang.cooling_purchase.provenance",
    "CoolingDatabaseEngine": "greenlang.cooling_purchase.cooling_database",
    "ElectricChillerCalculatorEngine": "greenlang.cooling_purchase.electric_chiller_calculator",
    "AbsorptionCoolingCalculatorEngine": "greenlang.cooling_purchase.absorption_cooling_calculator",
    "DistrictCoolingCalculatorEngine": "greenlang.cooling_purchase.district_cooling_calculator",
    "UncertaintyQuantifierEngine": "greenlang.cooling_purchase.uncertainty_quantifier",
    "ComplianceCheckerEngine": "greenlang.cooling_purchase.compliance_checker",
    "CoolingPurchasePipelineEngine": "greenlang.cooling_purchase.cooling_purchase_pipeline",
    "CoolingPurchaseService": "greenlang.cooling_purchase.setup",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module 'greenlang.cooling_purchase' has no attribute {name!r}")


__all__ = [
    "__version__",
    "__agent_id__",
    "__agent_label__",
] + list(_LAZY_IMPORTS.keys())
