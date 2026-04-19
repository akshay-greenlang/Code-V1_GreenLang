"""
Calculators for GL-015 INSULSCAN Agent

This module exports all calculator functions for insulation analysis.
All calculations are deterministic, following thermal engineering standards
with zero-hallucination principles.

Calculator Modules:
    - materials: 50+ insulation material database with thermal properties
    - heat_loss: Heat loss calculations (flat, cylindrical, IR-based)
    - economic_thickness: Economic thickness optimization and ROI analysis
"""

from .materials import (
    # Data classes
    InsulationMaterial,
    InsulationCategory,
    TemperatureClass,
    # Database access
    INSULATION_DATABASE,
    get_material,
    get_material_properties,
    list_materials,
    find_suitable_materials,
    get_material_count,
    get_categories,
    get_materials_by_category,
    recommend_material,
)

from .heat_loss import (
    # Data classes
    HeatLossResult,
    SURFACE_EMISSIVITY,
    # Coefficient calculations
    calculate_convection_coefficient,
    calculate_radiation_coefficient,
    calculate_surface_coefficient,
    # Heat loss calculations
    calculate_flat_surface_heat_loss,
    calculate_cylindrical_heat_loss,
    calculate_bare_surface_heat_loss,
    calculate_multilayer_heat_loss,
    calculate_heat_loss_savings,
    # IR integration
    estimate_heat_loss_from_ir_data,
    # Energy calculations
    calculate_annual_energy_loss,
    # Unit conversions
    convert_watts_to_btu_hr,
    convert_btu_hr_to_watts,
)

from .economic_thickness import (
    # Data classes
    EconomicThicknessResult,
    DEFAULT_PARAMS,
    # Economic calculations
    calculate_annualized_cost,
    calculate_npv,
    calculate_simple_payback,
    estimate_irr,
    calculate_insulation_cost,
    calculate_annual_energy_cost,
    # Economic thickness optimization
    calculate_economic_thickness_flat,
    calculate_economic_thickness_pipe,
    calculate_economic_thickness,
    # Comparison and analysis
    compare_materials_economically,
    calculate_roi_analysis,
)

__all__ = [
    # === Materials Module ===
    # Data classes
    "InsulationMaterial",
    "InsulationCategory",
    "TemperatureClass",
    # Database access
    "INSULATION_DATABASE",
    "get_material",
    "get_material_properties",
    "list_materials",
    "find_suitable_materials",
    "get_material_count",
    "get_categories",
    "get_materials_by_category",
    "recommend_material",

    # === Heat Loss Module ===
    # Data classes
    "HeatLossResult",
    "SURFACE_EMISSIVITY",
    # Coefficient calculations
    "calculate_convection_coefficient",
    "calculate_radiation_coefficient",
    "calculate_surface_coefficient",
    # Heat loss calculations
    "calculate_flat_surface_heat_loss",
    "calculate_cylindrical_heat_loss",
    "calculate_bare_surface_heat_loss",
    "calculate_multilayer_heat_loss",
    "calculate_heat_loss_savings",
    # IR integration
    "estimate_heat_loss_from_ir_data",
    # Energy calculations
    "calculate_annual_energy_loss",
    # Unit conversions
    "convert_watts_to_btu_hr",
    "convert_btu_hr_to_watts",

    # === Economic Thickness Module ===
    # Data classes
    "EconomicThicknessResult",
    "DEFAULT_PARAMS",
    # Economic calculations
    "calculate_annualized_cost",
    "calculate_npv",
    "calculate_simple_payback",
    "estimate_irr",
    "calculate_insulation_cost",
    "calculate_annual_energy_cost",
    # Economic thickness optimization
    "calculate_economic_thickness_flat",
    "calculate_economic_thickness_pipe",
    "calculate_economic_thickness",
    # Comparison and analysis
    "compare_materials_economically",
    "calculate_roi_analysis",
]
