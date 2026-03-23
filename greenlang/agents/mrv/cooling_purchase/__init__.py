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
    "CoolingTechnology": "greenlang.agents.mrv.cooling_purchase.models",
    "CompressorType": "greenlang.agents.mrv.cooling_purchase.models",
    "CondenserType": "greenlang.agents.mrv.cooling_purchase.models",
    "AbsorptionType": "greenlang.agents.mrv.cooling_purchase.models",
    "FreeCoolingSource": "greenlang.agents.mrv.cooling_purchase.models",
    "TESType": "greenlang.agents.mrv.cooling_purchase.models",
    "HeatSource": "greenlang.agents.mrv.cooling_purchase.models",
    "EfficiencyMetric": "greenlang.agents.mrv.cooling_purchase.models",
    "CoolingUnit": "greenlang.agents.mrv.cooling_purchase.models",
    "EmissionGas": "greenlang.agents.mrv.cooling_purchase.models",
    "GWPSource": "greenlang.agents.mrv.cooling_purchase.models",
    "ComplianceStatus": "greenlang.agents.mrv.cooling_purchase.models",
    "DataQualityTier": "greenlang.agents.mrv.cooling_purchase.models",
    "FacilityType": "greenlang.agents.mrv.cooling_purchase.models",
    "ReportingPeriod": "greenlang.agents.mrv.cooling_purchase.models",
    "AggregationType": "greenlang.agents.mrv.cooling_purchase.models",
    "BatchStatus": "greenlang.agents.mrv.cooling_purchase.models",
    "Refrigerant": "greenlang.agents.mrv.cooling_purchase.models",
    "ElectricChillerRequest": "greenlang.agents.mrv.cooling_purchase.models",
    "AbsorptionCoolingRequest": "greenlang.agents.mrv.cooling_purchase.models",
    "FreeCoolingRequest": "greenlang.agents.mrv.cooling_purchase.models",
    "TESRequest": "greenlang.agents.mrv.cooling_purchase.models",
    "DistrictCoolingRequest": "greenlang.agents.mrv.cooling_purchase.models",
    "CalculationResult": "greenlang.agents.mrv.cooling_purchase.models",
    "TESCalculationResult": "greenlang.agents.mrv.cooling_purchase.models",
    "BatchCalculationRequest": "greenlang.agents.mrv.cooling_purchase.models",
    "BatchCalculationResult": "greenlang.agents.mrv.cooling_purchase.models",
    "UncertaintyRequest": "greenlang.agents.mrv.cooling_purchase.models",
    "UncertaintyResult": "greenlang.agents.mrv.cooling_purchase.models",
    "ComplianceCheckResult": "greenlang.agents.mrv.cooling_purchase.models",
    "AggregationRequest": "greenlang.agents.mrv.cooling_purchase.models",
    "AggregationResult": "greenlang.agents.mrv.cooling_purchase.models",
    "CoolingPurchaseConfig": "greenlang.agents.mrv.cooling_purchase.config",
    "CoolingPurchaseMetrics": "greenlang.agents.mrv.cooling_purchase.metrics",
    "CoolingPurchaseProvenance": "greenlang.agents.mrv.cooling_purchase.provenance",
    "CoolingDatabaseEngine": "greenlang.agents.mrv.cooling_purchase.cooling_database",
    "ElectricChillerCalculatorEngine": "greenlang.agents.mrv.cooling_purchase.electric_chiller_calculator",
    "AbsorptionCoolingCalculatorEngine": "greenlang.agents.mrv.cooling_purchase.absorption_cooling_calculator",
    "DistrictCoolingCalculatorEngine": "greenlang.agents.mrv.cooling_purchase.district_cooling_calculator",
    "UncertaintyQuantifierEngine": "greenlang.agents.mrv.cooling_purchase.uncertainty_quantifier",
    "ComplianceCheckerEngine": "greenlang.agents.mrv.cooling_purchase.compliance_checker",
    "CoolingPurchasePipelineEngine": "greenlang.agents.mrv.cooling_purchase.cooling_purchase_pipeline",
    "CoolingPurchaseService": "greenlang.agents.mrv.cooling_purchase.setup",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module 'greenlang.agents.mrv.cooling_purchase' has no attribute {name!r}")


__all__ = [
    "__version__",
    "__agent_id__",
    "__agent_label__",
] + list(_LAZY_IMPORTS.keys())
